/*
 * S8MaliMetal command queue, command buffers and encoders.
 *
 * _MTLCommandQueue / _MTLCommandBuffer implement commit, enqueue, the
 * scheduled and completed handlers and waiting. The driver records each
 * encoder's work as a pass in its command buffer; -submitCommandBuffers:
 * count: runs the passes, reports them scheduled and, on a private serial
 * queue, completed (-commandBufferDidComplete:...).
 *
 * Stage 3: render passes run on the Mali (s8mtl_draw.c builds and submits
 * their job chains); blits still run on the CPU. A render pass the GPU path
 * cannot take (target not GPU-visible, unknown format, no shader cache)
 * falls back to stage 2: its clear runs on the CPU, its draws are skipped.
 */
#import "s8mtl.h"

void *memcpy(void *, const void *, size_t);
void *memset(void *, int, size_t);
void *realloc(void *, size_t);
void *malloc(size_t);
void free(void *);
void *dispatch_queue_create(const char *label, void *attr);
void dispatch_async_f(void *queue, void *context, void (*work)(void *));

/* ---- queue ---- */

struct s8_completion {
	S8CommandQueue *queue;
	id buffer;
	uint64_t start, end;
};

static void s8_complete(void *ctx)
{
	struct s8_completion *c = ctx;
	[c->queue commandBufferDidComplete:c->buffer startTime:c->start completionTime:c->end error:nil];
	[c->buffer release];
	[c->queue release];
	free(c);
}

@implementation S8CommandQueue

- (instancetype)initWithS8Device:(id)device
{
	self = [super initWithDevice:device];
	if (self == nil)
		return nil;
	_s8device = device;
	_completion = dispatch_queue_create("org.ios-galaxy-s8.mali.completion", NULL);
	return self;
}

S8_FORWARDING

- (id)device { return _s8device; }

- (id)commandBuffer
{
	S8CommandBuffer *cb = [[S8CommandBuffer alloc] initWithQueue:self retainedReferences:YES];
	if (cb != nil)
		cb->_s8device = _s8device;
	return [cb autorelease];
}

- (id)commandBufferWithUnretainedReferences
{
	return [self commandBuffer];
}

- (id)commandBufferWithDescriptor:(id)desc
{
	(void)desc;
	return [self commandBuffer];
}

- (BOOL)setGPUPriority:(NSUInteger)priority { (void)priority; return YES; }
- (BOOL)setBackgroundGPUPriority:(NSUInteger)priority { (void)priority; return YES; }
- (void)setCompletionQueue:(id)queue { _completionQueue = queue; }

- (void)submitCommandBuffers:(const id *)buffers count:(NSUInteger)count
{
	for (NSUInteger i = 0; i < count; i++) {
		S8CommandBuffer *cb = buffers[i];
		uint64_t start = s8_now_ns();
		[cb s8_execute];
		uint64_t end = s8_now_ns();
		[cb didScheduleWithStartTime:start endTime:start error:nil];
		struct s8_completion *c = malloc(sizeof(*c));
		c->queue = [self retain];
		c->buffer = [cb retain];
		c->start = start;
		c->end = end;
		dispatch_async_f(_completion, c, s8_complete);
	}
}

@end

/* ---- command buffer ---- */

static unsigned s8_frames, s8_draws_total;

@implementation S8CommandBuffer

- (unsigned)s8_newPass
{
	if (_npasses == _cap) {
		_cap = _cap ? _cap * 2 : 8;
		_passes = realloc(_passes, _cap * sizeof(*_passes));
	}
	memset(&_passes[_npasses], 0, sizeof(*_passes));
	return _npasses++;
}

- (struct s8_pass *)s8_pass:(unsigned)index
{
	return &_passes[index];
}

- (struct s8d_cmdbuf *)s8_draw
{
	if (_draw == NULL && s8d_init() == 0)
		_draw = s8d_cmdbuf_new();
	return _draw;
}

- (void)dealloc
{
	s8d_cmdbuf_free(_draw);
	for (unsigned i = 0; i < _npasses; i++) {
		[_passes[i].target release];
		[_passes[i].src release];
		[_passes[i].dst release];
	}
	free(_passes);
	[super dealloc];
}

S8_FORWARDING

- (id)device { return _s8device; }
- (NSUInteger)protectionOptions { return 0; }
- (void)setProtectionOptions:(NSUInteger)options { (void)options; }

- (id)renderCommandEncoderWithDescriptor:(id)desc
{
	S8RenderEncoder *e = [[S8RenderEncoder alloc] initWithCommandBuffer:self descriptor:desc];
	[self setCurrentCommandEncoder:e];
	return [e autorelease];
}

- (id)blitCommandEncoder
{
	S8BlitEncoder *e = [[S8BlitEncoder alloc] initWithS8CommandBuffer:self];
	[self setCurrentCommandEncoder:e];
	return [e autorelease];
}

- (id)blitCommandEncoderWithDescriptor:(id)desc
{
	(void)desc;
	return [self blitCommandEncoder];
}

- (id)computeCommandEncoder
{
	S8ComputeEncoder *e = [[S8ComputeEncoder alloc] initWithCommandBuffer:self];
	[self setCurrentCommandEncoder:e];
	return [e autorelease];
}

- (id)computeCommandEncoderWithDispatchType:(NSUInteger)type
{
	(void)type;
	return [self computeCommandEncoder];
}

- (id)computeCommandEncoderWithDescriptor:(id)desc
{
	(void)desc;
	return [self computeCommandEncoder];
}

/* Clear colour packed for the texture's format (8-bit unorm formats). */
static uint32_t s8_pack(NSUInteger fmt, MTLClearColor c)
{
	double ch[4] = { c.red, c.green, c.blue, c.alpha };
	uint32_t b[4];
	for (int i = 0; i < 4; i++) {
		double v = ch[i] < 0 ? 0 : (ch[i] > 1 ? 1 : ch[i]);
		b[i] = (uint32_t)(v * 255.0 + 0.5);
	}
	if (fmt == MTLPixelFormatBGRA8Unorm || fmt == MTLPixelFormatBGRA8Unorm_sRGB)
		return b[2] | b[1] << 8 | b[0] << 16 | b[3] << 24;
	return b[0] | b[1] << 8 | b[2] << 16 | b[3] << 24;
}

static void s8_cpu_clear(S8Texture *t, MTLClearColor c)
{
	if (t == nil || t->_cpu == NULL)
		return;
	unsigned bpp = s8_format_bpp(t->_format);
	if (bpp == 4) {
		uint32_t v = s8_pack(t->_format, c);
		for (NSUInteger y = 0; y < t->_height; y++) {
			uint32_t *row = (uint32_t *)(t->_cpu + y * t->_rowBytes);
			for (NSUInteger x = 0; x < t->_width; x++)
				row[x] = v;
		}
	} else {
		for (NSUInteger y = 0; y < t->_height; y++)
			memset(t->_cpu + y * t->_rowBytes, 0, t->_width * bpp);
	}
}

/*
 * Screen thumbnails in the log (test/shot2png.py makes PNGs of them): the
 * full-screen BGRA targets at 20, 60 and 150 s after the first one, apart
 * for the display surface (format 554, shots 1-3; boot shot1 showed it blue)
 * and the 550 surfaces CoreAnimation renders into first (shots 11-13; black
 * in shot1). 16x16 blocks averaged over 4x4 samples, 6 bits per channel,
 * base64 (3 characters a pixel), 30 pixels a line.
 */
/*
 * Diagnostic windows: the first 3 seconds of every minute from the first
 * command buffer on (1:00, 2:00, ...), up to 250 lines each, every texture
 * binding and render pass is logged, so what an app shows at that moment
 * (Settings' text breaking on refresh) can be read back.
 */
/* Boot diag2's minute windows caught nothing (iOS draws ~1.5 frames a
 * second when idle): instead, the first command buffer after every 20 s
 * is logged whole (bindings while it is encoded, passes when it runs). */
static uint64_t s8_diag_next;
static BOOL s8_diag_on;
static unsigned s8_diag_lines, s8_diag_n;

static BOOL s8_diag(void)
{
	uint64_t now = s8_now_ns();
	if (s8_diag_next == 0)
		s8_diag_next = now + 30000000000ULL;
	if (!s8_diag_on && now >= s8_diag_next && s8_diag_n < 30) {
		s8_diag_on = YES;
		s8_diag_lines = 0;
		s8_log("MTL diag snapshot %u", ++s8_diag_n);
	}
	return s8_diag_on && s8_diag_lines++ < 200;
}

static void s8_diag_done(void)
{
	if (s8_diag_on) {
		s8_diag_on = NO;
		s8_diag_next = s8_now_ns() + 20000000000ULL;
	}
}

/* IOSurfaces mapped for the command buffer being encoded (the kernel has
 * 16 windows): more than 16 distinct ones are logged when it runs. */
static uint32_t s8_cb_surf[64];
static unsigned s8_cb_nsurf;

void s8_note_surface(uint32_t id)
{
	for (unsigned i = 0; i < s8_cb_nsurf; i++)
		if (s8_cb_surf[i] == id)
			return;
	if (s8_cb_nsurf < 64)
		s8_cb_surf[s8_cb_nsurf++] = id;
}

static void s8_shot(S8Texture *t)
{
	static uint64_t t0;
	static unsigned taken[2];
	static const unsigned at[] = { 61, 121, 181 };   /* with the diagnostic windows */
	static const char b64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	if (t == nil || t->_cpu == NULL || t->_width < 1000 || t->_height < 2000 || s8_format_bpp(t->_format) != 4)
		return;
	unsigned slot = t->_format == 554 ? 0 : 1;
	if (taken[slot] >= sizeof(at) / sizeof(at[0]))
		return;
	uint64_t now = s8_now_ns();
	if (t0 == 0)
		t0 = now;
	if (now - t0 < (uint64_t)at[taken[slot]] * 1000000000ULL)
		return;
	unsigned shots = ++taken[slot] + (slot ? 10 : 0);
	unsigned bw = (unsigned)t->_width / 16, bh = (unsigned)t->_height / 16;
	NSUInteger rb = t->_rowBytes ? t->_rowBytes : t->_width * 4;
	s8_log("MTL shot %u: %ux%u of %lux%lu fmt %lu at %llu s", shots, bw, bh, (unsigned long)t->_width,
	    (unsigned long)t->_height, (unsigned long)t->_format, (unsigned long long)((now - t0) / 1000000000ULL));
	for (unsigned by = 0; by < bh; by++) {
		char line[100];
		unsigned n = 0, part = 0;
		for (unsigned bx = 0; bx < bw; bx++) {
			unsigned sum[3] = { 0, 0, 0 };
			for (unsigned sy = 0; sy < 4; sy++)
				for (unsigned sx = 0; sx < 4; sx++) {
					const uint8_t *px = t->_cpu + (by * 16 + sy * 4 + 2) * rb + (bx * 16 + sx * 4 + 2) * 4;
					sum[0] += px[2];        /* BGRA in memory */
					sum[1] += px[1];
					sum[2] += px[0];
				}
			for (int c = 0; c < 3; c++)
				line[n++] = b64[(sum[c] / 16) >> 2];
			if (n == 90 || bx + 1 == bw) {
				line[n] = 0;
				s8_log("MTL shot %u %u %u: %s", shots, by, part++, line);
				n = 0;
			}
		}
	}
}

void s8_shadow_refresh(void);

- (void)s8_execute
{
	unsigned draws = 0, clears = 0, blits = 0, gpu = 0;
	/* copied textures again, with what was written since they were bound */
	s8_shadow_refresh();
	for (unsigned i = 0; i < _npasses; i++) {
		struct s8_pass *p = &_passes[i];
		if (p->kind == 1) {
			draws += p->draws;
			if (p->gpu >= 0 && s8d_pass_execute(_draw, p->gpu) == 0) {
				gpu++;
				s8_shot(p->target);
				/* where the colours go (boot tex1: the display only blue):
				 * two raw texels of each of the first 80 full-screen
				 * passes, centre and lower left */
				S8Texture *tg = p->target;
				static unsigned probed;
				if (tg != nil && tg->_cpu != NULL && tg->_rowBytes && tg->_width >= 1000 &&
				    s8_format_bpp(tg->_format) == 4 && probed++ < 80) {
					uint32_t a, b;
					memcpy(&a, tg->_cpu + (tg->_height / 2) * tg->_rowBytes + (tg->_width / 2) * 4, 4);
					memcpy(&b, tg->_cpu + (tg->_height * 9 / 10) * tg->_rowBytes + (tg->_width / 8) * 4, 4);
					s8_log("MTL probe %lux%lu fmt %lu load %lu draws %u: %08x %08x", (unsigned long)tg->_width,
					    (unsigned long)tg->_height, (unsigned long)tg->_format, (unsigned long)p->load,
					    p->draws, a, b);
				}
			} else if (p->load == MTLLoadActionClear) {
				s8_cpu_clear(p->target, p->clear);
				clears++;
			}
		} else if (p->kind == 2) {
			S8Buffer *s = p->src, *d = p->dst;
			if (s->_mem.cpu != NULL && d->_mem.cpu != NULL)
				memcpy((uint8_t *)d->_mem.cpu + p->dstOff, (uint8_t *)s->_mem.cpu + p->srcOff, p->size);
			blits++;
		} else if (p->kind == 3) {
			S8Texture *s = p->src, *d = p->dst;
			unsigned bpp = s8_format_bpp(s->_format);
			if (s->_cpu != NULL && d->_cpu != NULL)
				for (NSUInteger y = 0; y < p->sz.height; y++)
					memcpy(d->_cpu + (p->dO.y + y) * d->_rowBytes + p->dO.x * bpp,
					    s->_cpu + (p->so.y + y) * s->_rowBytes + p->so.x * bpp, p->sz.width * bpp);
			blits++;
		}
	}
	s8_draws_total += draws;
	if (s8_diag()) {
		for (unsigned i = 0; i < _npasses; i++) {
			struct s8_pass *p = &_passes[i];
			S8Texture *tg = p->target;
			s8_log("MTL diag pass %u/%u kind %d gpu %d: %lux%lu f%lu load %lu draws %u", i, _npasses, p->kind, p->gpu,
			    tg ? (unsigned long)tg->_width : 0, tg ? (unsigned long)tg->_height : 0,
			    tg ? (unsigned long)tg->_format : 0, (unsigned long)p->load, p->draws);
		}
		s8_log("MTL diag cb: %u passes, %u IOSurfaces mapped", _npasses, s8_cb_nsurf);
	}
	s8_diag_done();
	{
		static unsigned over_logged;
		if (s8_cb_nsurf > 16 && over_logged++ < 60)
			s8_log("MTL cb: %u IOSurfaces mapped (> 16 kernel windows), %u passes, %u draws", s8_cb_nsurf,
			    _npasses, draws);
		s8_cb_nsurf = 0;
	}
	if (++s8_frames % 120 == 1)
		s8_log("MTL command buffer %u: %u passes (%u on the GPU), %u CPU clears, %u blits, %u draws (%u so far)",
		    s8_frames, _npasses, gpu, clears, blits, draws, s8_draws_total);
}

@end

/* ---- render encoder ---- */

uint32_t IOSurfaceGetID(void *surface);

/*
 * What the fragment shaders sample (boot idx1: every image layer composited
 * black): the first 300 fragment bindings and every failed one (up to
 * 100), with the texture's class, size, format, GPU address, IOSurface,
 * parent (views) and the first and centre texels as the CPU sees them.
 */
static void s8_log_binding(unsigned stage, NSUInteger index, id t, int ok, const struct s8d_texture *ref)
{
	static unsigned logged, failed;
	if (stage != S8D_STAGE_FRAGMENT || t == nil)
		return;
	if ((ok ? logged++ >= 300 : failed++ >= 100) && !s8_diag())
		return;
	if (![t isKindOfClass:[S8Texture class]]) {
		s8_log("MTL bind fs[%lu]: foreign %s", (unsigned long)index, class_getName(object_getClass(t)));
		return;
	}
	S8Texture *x = t;
	uint32_t first = 0, centre = 0;
	unsigned bpp = s8_format_bpp(x->_format);
	if (x->_cpu != NULL && x->_rowBytes && bpp && x->_width >= 2) {
		memcpy(&first, x->_cpu, 4);
		memcpy(&centre, x->_cpu + (x->_height / 2) * x->_rowBytes + (x->_width / 2) * bpp, 4);
	}
	s8_log("MTL bind fs[%lu]%s: %lux%lu f%lu rb %lu u%lx gpu %llx s%u%s px %08x/%08x", (unsigned long)index,
	    ok ? "" : " NULL", (unsigned long)x->_width, (unsigned long)x->_height, (unsigned long)x->_format,
	    (unsigned long)x->_rowBytes, (unsigned long)x->_usage,
	    (unsigned long long)(ok ? ref->gpu : x->_gpu), x->_iosurface ? IOSurfaceGetID(x->_iosurface) : 0,
	    x->_parent != nil ? " v" : "", first, centre);
}

static void s8_bind_texture(struct s8d_cmdbuf *d, unsigned stage, NSUInteger index, id t)
{
	struct s8d_texture ref;
	if (d == NULL)
		return;
	int ok = t != nil && s8_texture_ref(t, &ref) == 0;
	s8_log_binding(stage, index, t, ok, &ref);
	if (ok)
		s8d_set_texture(d, stage, (unsigned)index, &ref);
	else
		s8d_set_texture(d, stage, (unsigned)index, NULL);
}

static void s8_bind_buffer(struct s8d_cmdbuf *d, unsigned stage, NSUInteger index, id b, NSUInteger offset)
{
	S8Buffer *buf = b;
	if (d == NULL)
		return;
	if (buf == nil) {
		s8d_set_buffer(d, stage, (unsigned)index, 0, NULL, 0);
		return;
	}
	uint64_t len = buf->_length > offset ? buf->_length - offset : 0;
	s8d_set_buffer(d, stage, (unsigned)index, buf->_mem.gpu && buf->_nocopy == NULL ? buf->_mem.gpu + offset : 0,
	    buf->_mem.cpu ? (uint8_t *)buf->_mem.cpu + offset : NULL, len);
}

@implementation S8RenderEncoder

- (instancetype)initWithCommandBuffer:(S8CommandBuffer *)cb descriptor:(id)desc
{
	self = [super initWithCommandBuffer:cb];
	if (self == nil)
		return nil;
	_cb = cb;
	_pass = [cb s8_newPass];
	struct s8_pass *p = [cb s8_pass:_pass];
	id ca = [[desc colorAttachments] objectAtIndexedSubscript:0];
	p->kind = 1;
	p->gpu = -1;
	p->target = [[ca texture] retain];
	p->load = [ca loadAction];
	p->clear = [ca clearColor];
	struct s8d_cmdbuf *d = [cb s8_draw];
	struct s8d_texture ref;
	if (d != NULL && p->target != nil && s8_texture_ref(p->target, &ref) == 0) {
		double c[4] = { p->clear.red, p->clear.green, p->clear.blue, p->clear.alpha };
		p->gpu = s8d_pass_begin(d, &ref, (unsigned)p->load, c);
	}
	return self;
}

S8_FORWARDING

- (struct s8_pass *)p { return [_cb s8_pass:_pass]; }
- (struct s8d_cmdbuf *)d { return [self p]->gpu >= 0 ? _cb->_draw : NULL; }

- (void)endEncoding
{
	if ([self d] != NULL)
		s8d_pass_end([self d]);
	[super endEncoding];
}

- (void)setRenderPipelineState:(id)state
{
	S8RenderPipelineState *ps = state;
	[self p]->pipelines++;
	if ([self d] != NULL)
		s8d_set_pipeline([self d], ps ? ps->_gpu : NULL);
}
- (void)setDepthStencilState:(id)state { (void)state; }
/* Buffers stay bound (not retained) per stage so setXBufferOffset: can
 * rebind them (boot 52: backboardd changes offsets between draws). */
- (void)s8_buffer:(unsigned)stage index:(NSUInteger)i buffer:(id)b offset:(NSUInteger)o
{
	if (i >= S8D_MAX_BUFFERS)
		return;
	_buffers[stage][i] = b;
	s8_bind_buffer([self d], stage, i, b, o);
}
- (void)setVertexBuffer:(id)b offset:(NSUInteger)o atIndex:(NSUInteger)i
{
	[self s8_buffer:S8D_STAGE_VERTEX index:i buffer:b offset:o];
}
- (void)setVertexBuffers:(const id *)b offsets:(const NSUInteger *)o withRange:(NSRange)r
{
	for (NSUInteger k = 0; k < r.length; k++)
		[self s8_buffer:S8D_STAGE_VERTEX index:r.location + k buffer:b[k] offset:o[k]];
}
- (void)setVertexBufferOffset:(NSUInteger)o atIndex:(NSUInteger)i
{
	if (i < S8D_MAX_BUFFERS)
		s8_bind_buffer([self d], S8D_STAGE_VERTEX, i, _buffers[S8D_STAGE_VERTEX][i], o);
}
- (void)setVertexBytes:(const void *)p length:(NSUInteger)l atIndex:(NSUInteger)i
{
	if (i < S8D_MAX_BUFFERS)
		_buffers[S8D_STAGE_VERTEX][i] = nil;
	if ([self d] != NULL)
		s8d_set_bytes([self d], S8D_STAGE_VERTEX, (unsigned)i, p, l);
}
- (void)setVertexTexture:(id)t atIndex:(NSUInteger)i { s8_bind_texture([self d], S8D_STAGE_VERTEX, i, t); }
- (void)setVertexSamplerState:(id)s atIndex:(NSUInteger)i
{
	S8SamplerState *ss = s;
	if ([self d] != NULL)
		s8d_set_sampler([self d], S8D_STAGE_VERTEX, (unsigned)i, ss ? &ss->_s8 : NULL);
}
- (void)setFragmentBuffer:(id)b offset:(NSUInteger)o atIndex:(NSUInteger)i
{
	[self s8_buffer:S8D_STAGE_FRAGMENT index:i buffer:b offset:o];
}
- (void)setFragmentBuffers:(const id *)b offsets:(const NSUInteger *)o withRange:(NSRange)r
{
	for (NSUInteger k = 0; k < r.length; k++)
		[self s8_buffer:S8D_STAGE_FRAGMENT index:r.location + k buffer:b[k] offset:o[k]];
}
- (void)setFragmentBufferOffset:(NSUInteger)o atIndex:(NSUInteger)i
{
	if (i < S8D_MAX_BUFFERS)
		s8_bind_buffer([self d], S8D_STAGE_FRAGMENT, i, _buffers[S8D_STAGE_FRAGMENT][i], o);
}
- (void)setFragmentBytes:(const void *)p length:(NSUInteger)l atIndex:(NSUInteger)i
{
	if (i < S8D_MAX_BUFFERS)
		_buffers[S8D_STAGE_FRAGMENT][i] = nil;
	if ([self d] != NULL)
		s8d_set_bytes([self d], S8D_STAGE_FRAGMENT, (unsigned)i, p, l);
}
- (void)setFragmentTexture:(id)t atIndex:(NSUInteger)i { s8_bind_texture([self d], S8D_STAGE_FRAGMENT, i, t); }
- (void)setFragmentTextures:(const id *)t withRange:(NSRange)r
{
	for (NSUInteger k = 0; k < r.length; k++)
		s8_bind_texture([self d], S8D_STAGE_FRAGMENT, r.location + k, t[k]);
}
- (void)setFragmentSamplerState:(id)s atIndex:(NSUInteger)i
{
	S8SamplerState *ss = s;
	if ([self d] != NULL)
		s8d_set_sampler([self d], S8D_STAGE_FRAGMENT, (unsigned)i, ss ? &ss->_s8 : NULL);
}
- (void)setFragmentSamplerStates:(const id *)s withRange:(NSRange)r
{
	for (NSUInteger k = 0; k < r.length; k++)
		[self setFragmentSamplerState:s[k] atIndex:r.location + k];
}
- (void)setFragmentTexture:(id)t atTextureIndex:(NSUInteger)ti samplerState:(id)s atSamplerIndex:(NSUInteger)si
{
	[self setFragmentTexture:t atIndex:ti];
	[self setFragmentSamplerState:s atIndex:si];
}
- (void)setViewport:(MTLViewport)v
{
	if ([self d] != NULL)
		s8d_set_viewport([self d], v.originX, v.originY, v.width, v.height, v.znear, v.zfar);
}
- (void)setScissorRect:(MTLScissorRect)r
{
	if ([self d] != NULL)
		s8d_set_scissor([self d], (unsigned)r.x, (unsigned)r.y, (unsigned)r.width, (unsigned)r.height);
}
- (void)setCullMode:(NSUInteger)m { _cull = m; if ([self d] != NULL) s8d_set_cull([self d], (unsigned)_cull, (unsigned)_winding); }
- (void)setFrontFacingWinding:(NSUInteger)w { _winding = w; if ([self d] != NULL) s8d_set_cull([self d], (unsigned)_cull, (unsigned)_winding); }
- (void)setTriangleFillMode:(NSUInteger)m { (void)m; }
- (void)setDepthBias:(float)b slopeScale:(float)s clamp:(float)c { (void)b; (void)s; (void)c; }
- (void)setStencilReferenceValue:(uint32_t)v { (void)v; }
- (void)setBlendColorRed:(float)r green:(float)g blue:(float)b alpha:(float)a
{
	if ([self d] != NULL)
		s8d_set_blend_color([self d], r, g, b, a);
}
- (void)setVisibilityResultMode:(NSUInteger)m offset:(NSUInteger)o { (void)m; (void)o; }
- (void)setTileBytes:(const void *)p length:(NSUInteger)l atIndex:(NSUInteger)i { (void)p; (void)l; (void)i; }
- (void)setTileTexture:(id)t atIndex:(NSUInteger)i { (void)t; (void)i; }
- (void)drawPrimitives:(NSUInteger)type vertexStart:(NSUInteger)s vertexCount:(NSUInteger)c
{
	[self drawPrimitives:type vertexStart:s vertexCount:c instanceCount:1];
}
- (void)drawPrimitives:(NSUInteger)type vertexStart:(NSUInteger)s vertexCount:(NSUInteger)c
    instanceCount:(NSUInteger)n
{
	[self p]->draws++;
	if ([self d] != NULL)
		s8d_draw([self d], (unsigned)type, (unsigned)s, (unsigned)c, (unsigned)n, 0, 0, NULL);
}
- (void)drawIndexedPrimitives:(NSUInteger)type indexCount:(NSUInteger)c indexType:(NSUInteger)t
    indexBuffer:(id)b indexBufferOffset:(NSUInteger)o instanceCount:(NSUInteger)n
{
	S8Buffer *buf = b;
	[self p]->draws++;
	if ([self d] == NULL || buf == nil)
		return;
	unsigned size = t == 1 ? 4 : 2;   /* MTLIndexTypeUInt32 = 1 */
	const uint8_t *cpu = buf->_mem.cpu ? (const uint8_t *)buf->_mem.cpu + o : NULL;
	uint64_t gpu = buf->_mem.gpu && buf->_nocopy == NULL ? buf->_mem.gpu + o : 0;
	s8d_draw([self d], (unsigned)type, 0, (unsigned)c, (unsigned)n, size, gpu, cpu);
}
- (void)drawIndexedPrimitives:(NSUInteger)type indexCount:(NSUInteger)c indexType:(NSUInteger)t
    indexBuffer:(id)b indexBufferOffset:(NSUInteger)o
{
	[self drawIndexedPrimitives:type indexCount:c indexType:t indexBuffer:b indexBufferOffset:o instanceCount:1];
}

@end

/* ---- blit encoder ---- */

@implementation S8BlitEncoder

- (instancetype)initWithS8CommandBuffer:(S8CommandBuffer *)cb
{
	self = [super initWithCommandBuffer:cb];
	if (self != nil)
		_cb = cb;
	return self;
}

S8_FORWARDING

- (void)copyFromBuffer:(id)src sourceOffset:(NSUInteger)so toBuffer:(id)dst destinationOffset:(NSUInteger)d
    size:(NSUInteger)size
{
	struct s8_pass *p = [_cb s8_pass:[_cb s8_newPass]];
	p->kind = 2;
	p->src = [src retain];
	p->dst = [dst retain];
	p->srcOff = so;
	p->dstOff = d;
	p->size = size;
}

- (void)copyFromTexture:(id)src sourceSlice:(NSUInteger)ss sourceLevel:(NSUInteger)sl
    sourceOrigin:(MTLOrigin)so sourceSize:(MTLSize)size toTexture:(id)dst
    destinationSlice:(NSUInteger)ds destinationLevel:(NSUInteger)dl destinationOrigin:(MTLOrigin)dO
{
	(void)ss; (void)sl; (void)ds; (void)dl;
	struct s8_pass *p = [_cb s8_pass:[_cb s8_newPass]];
	p->kind = 3;
	p->src = [src retain];
	p->dst = [dst retain];
	p->so = so;
	p->dO = dO;
	p->sz = size;
}

- (void)generateMipmapsForTexture:(id)t { (void)t; }
- (void)synchronizeResource:(id)r { (void)r; }
- (void)fillBuffer:(id)b range:(NSRange)r value:(uint8_t)v
{
	S8Buffer *buf = b;
	if (buf->_mem.cpu != NULL)
		memset((uint8_t *)buf->_mem.cpu + r.location, v, r.length);
}

@end

/* ---- compute encoder (stage 2: dispatches skipped) ---- */

const char *getprogname(void);

/* Which kernels would run and on what (the skipped work is what stays
 * black or stale on screen): the first 40 dispatches, with the textures
 * bound since the last one. */
static unsigned s8_dispatches;

static void s8_log_dispatch(id pipeline, const char *how, MTLSize g, MTLSize t)
{
	if (s8_dispatches++ >= 40)
		return;
	S8ComputePipelineState *p = pipeline;
	s8_log("MTL dispatch %.40s %s %lux%lux%lu x %lux%lux%lu [%s]", p ? p->_name : "-", how,
	    (unsigned long)g.width, (unsigned long)g.height, (unsigned long)g.depth, (unsigned long)t.width,
	    (unsigned long)t.height, (unsigned long)t.depth, getprogname());
}

@implementation S8ComputeEncoder
S8_FORWARDING
- (void)setComputePipelineState:(id)s { _pipeline = s; }
- (void)setBuffer:(id)b offset:(NSUInteger)o atIndex:(NSUInteger)i { (void)b; (void)o; (void)i; }
- (void)setTexture:(id)t atIndex:(NSUInteger)i
{
	if (s8_dispatches < 40 && t != nil && [t isKindOfClass:[S8Texture class]]) {
		S8Texture *x = t;
		s8_log("MTL compute texture %lu: %lux%lu fmt %lu surf %u usage 0x%lx", (unsigned long)i,
		    (unsigned long)x->_width, (unsigned long)x->_height, (unsigned long)x->_format,
		    x->_iosurface ? IOSurfaceGetID(x->_iosurface) : 0, (unsigned long)x->_usage);
	}
}
- (void)setThreadgroupMemoryLength:(NSUInteger)l atIndex:(NSUInteger)i { (void)l; (void)i; }
- (void)dispatchThreadgroups:(MTLSize)g threadsPerThreadgroup:(MTLSize)t { s8_log_dispatch(_pipeline, "groups", g, t); }
- (void)dispatchThreads:(MTLSize)g threadsPerThreadgroup:(MTLSize)t { s8_log_dispatch(_pipeline, "threads", g, t); }
@end
