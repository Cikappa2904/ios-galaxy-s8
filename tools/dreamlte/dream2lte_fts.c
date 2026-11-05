/*
 * Galaxy S8+ (SM-G955F, dream2lte): the touch controller is an STMicro
 * FTS8CD56 at 0x49 on the same hsi2c_23 bus, not the S8's S6SY761 (stock
 * dream2lte DT: hsi2c@108E0000/touchscreen@49, "stm,fts_touch", rails
 * LDO34 tsp_io 1.85 V / LDO35 tsp_avdd 3.3 V). Protocol from Samsung's
 * drivers/input/touchscreen/stm/fts_ts.c (driver 0132, the version on the
 * phone): "D3" registers are written as B6 00 <reg> [value]; a register
 * read returns one dummy byte before the data. Polled through the event
 * count register, so the INT line is not used.
 */
#define S8FTS_ADDR              0x49u
#define S8FTS_EV_READY          0x10u   /* EVENTID_CONTROLLER_READY */
#define S8FTS_EV_ENTER          0x03u   /* EVENTID_ENTER_POINTER */
#define S8FTS_EV_LEAVE          0x04u   /* EVENTID_LEAVE_POINTER */
#define S8FTS_EV_MOTION         0x05u   /* EVENTID_MOTION_POINTER */
#define S8FTS_FIFO_MAX          31u     /* 8-byte events per read (HSI2C length) */

static bool
s8fts_xfer(const uint8_t *w, unsigned wlen, uint8_t *r, unsigned rlen)
{
	return s8i2c_xfer(s8ts_i2c, S8FTS_ADDR, w, wlen, r, rlen);
}

static bool
s8fts_cmd(uint8_t cmd)
{
	return s8fts_xfer(&cmd, 1, NULL, 0);
}

/* Panel pixels (0..1439, 0..2959) -> SweetHIDTouch's 1080x1920 grid. */
static void
s8fts_report(const uint8_t *ev)
{
	unsigned id = ev[0] & 0x0fu;
	unsigned tid = ev[0] >> 4;
	if (id < S8FTS_EV_ENTER || id > S8FTS_EV_MOTION || tid >= S8TS_FINGERS) {
		return;         /* status, release info, gesture, ... */
	}
	unsigned x = ev[1] | ((unsigned)(ev[2] & 0x0fu) << 8);
	unsigned y = (unsigned)(ev[2] >> 4) | ((unsigned)ev[3] << 4);
	x = x > 1439 ? 1439 : x;
	y = y > 2959 ? 2959 : y;
	if (s8ts_native) {
		x = x * 1080 / 1440;
		y = y * 1920 / 2960;
	} else {
		/* 1125x2436 is shown centered on the 1440x2960 panel. */
		int px = (int)x - (1440 - 1125) / 2;
		int py = (int)y - (2960 - 2436) / 2;
		px = px < 0 ? 0 : (px > 1124 ? 1124 : px);
		py = py < 0 ? 0 : (py > 2435 ? 2435 : py);
		x = (unsigned)px * 1080 / 1125;
		y = (unsigned)py * 1920 / 2436;
	}
	if (id == S8FTS_EV_ENTER || id == S8FTS_EV_MOTION) {
		if (!s8ts_down[tid]) {
			printf("S8+ TOUCH: down id=%u x=%u y=%u\n", tid, x, y);
		}
		s8ts_down[tid] = true;
		s8ts_x[tid] = (uint16_t)x;
		s8ts_y[tid] = (uint16_t)y;
		if (tid < 5) {
			sweet_hid_touch((int)tid, 1, (int)x, (int)y);
		}
	} else if (s8ts_down[tid]) {                /* S8FTS_EV_LEAVE */
		s8ts_down[tid] = false;
		if (tid < 5) {
			sweet_hid_touch((int)tid, 0, s8ts_x[tid], s8ts_y[tid]);
		}
	}
}

static void
s8fts_thread(void *arg, wait_result_t result)
{
	static uint8_t buf[8 * S8FTS_FIFO_MAX];
	const uint8_t count_reg[3] = {0xb6, 0x00, 0x23};
	const uint8_t read_all = 0x86;                  /* READ_ALL_EVENT */
	unsigned errors = 0;
	(void)arg;
	(void)result;
	printf("S8+ TOUCH: FTS worker started\n");
	for (;;) {
		uint8_t c[2] = {0, 0};
		if (!s8fts_xfer(count_reg, 3, c, 2)) {
			if (++errors <= 8 || !(errors & 255)) {
				printf("S8+ TOUCH: event count read failed (%u)\n", errors);
			}
			s8ts_wait(20);
			continue;
		}
		unsigned n = c[1] / 2u;                 /* fts_interrupt_handler() */
		if (n == 0) {
			s8ts_wait(8);
			continue;
		}
		n = n > S8FTS_FIFO_MAX ? S8FTS_FIFO_MAX : n;
		if (!s8fts_xfer(&read_all, 1, buf, n * 8)) {
			s8ts_wait(8);
			continue;
		}
		for (unsigned i = 0; i < n; i++) {
			s8fts_report(buf + i * 8);
		}
		errors = 0;
	}
}

/* fts_init(): system reset, wait for READY, chip id, sense on, INT on. */
static bool
s8fts_start(void)
{
	const uint8_t crc_en[4] = {0xb6, 0x00, 0x1e, 0x20};   /* warm boot CRC */
	const uint8_t reset[4] = {0xb6, 0x00, 0x28, 0x80};    /* system reset */
	const uint8_t chip_reg[3] = {0xb6, 0x00, 0x04};
	const uint8_t int_on[4] = {0xb6, 0x00, 0x2c, 0x48};   /* INT_ENABLE_D3 */
	const uint8_t read_one = 0x85;                        /* READ_ONE_EVENT */
	uint8_t ev[8] = {0}, id[7] = {0};
	thread_t thread;
	bool ready = false;

	(void)s8fts_xfer(crc_en, 4, NULL, 0);
	delay(10 * 1000);
	if (!s8fts_xfer(reset, 4, NULL, 0)) {
		printf("S8+ TOUCH: no FTS at %#x either\n", S8FTS_ADDR);
		return false;
	}
	delay(10 * 1000);
	for (unsigned i = 0; i < 20 && !ready; i++) {
		if (s8fts_xfer(&read_one, 1, ev, 8) && ev[0] == S8FTS_EV_READY) {
			ready = true;
		} else {
			delay(20 * 1000);
		}
	}
	printf("S8+ TOUCH: FTS reset %s, event %02x %02x %02x %02x %02x %02x %02x %02x\n",
	    ready ? "ready" : "timeout", ev[0], ev[1], ev[2], ev[3], ev[4], ev[5], ev[6], ev[7]);
	if (!s8fts_xfer(chip_reg, 3, id, 7) || id[1] != 0x36 || id[2] != 0x70) {
		printf("S8+ TOUCH: FTS chip id %02x %02x %02x %02x %02x %02x (want 36 70)\n",
		    id[1], id[2], id[3], id[4], id[5], id[6]);
		return false;
	}
	printf("S8+ TOUCH: FTS8CD56 id %02x %02x, fw %02x%02x\n", id[1], id[2], id[6], id[5]);
	(void)s8fts_cmd(0xa1);                          /* FLUSHBUFFER */
	delay(10 * 1000);
	if (!s8fts_cmd(0x93)) {                         /* SENSEON */
		printf("S8+ TOUCH: FTS sense on failed\n");
		return false;
	}
	(void)s8fts_cmd(0xa2);                          /* FORCECALIBRATION */
	(void)s8fts_xfer(int_on, 4, NULL, 0);
	if (kernel_thread_start(s8fts_thread, NULL, &thread) != KERN_SUCCESS) {
		printf("S8+ TOUCH: thread start failed\n");
		return false;
	}
	thread_deallocate(thread);
	return true;
}

