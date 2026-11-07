/*
 * S8MaliMetal.bundle: Metal on the Galaxy S8's Mali-G71 for iOS.
 *
 * Metal.framework (MTLRegisterDevices) matches every IOAcceleratorES
 * service, loads /System/Library/Extensions/<MetalPluginName>.bundle and
 * instantiates <MetalPluginClassName>, which must subclass Metal's
 * _MTLDevice and conform to MTLDevice (boot 47), with
 * -initWithAcceleratorPort:. The S8 kernel publishes such a service with
 * boot-arg s8metal=1 (S8MaliAccelerator in SweetFramebuffer.cpp); the GPU
 * itself is reached through /dev/s8mali (tools/dreamlte/s8mali_ioctl.h).
 *
 * S8MaliDevice (stage 2):
 *
 * _MTLDevice supplies limits, GPU family answers and library parsing; this
 * class creates the driver's objects. It presents itself like the A11
 * family (featureProfile 5) but without tile shaders, memoryless targets,
 * raster order groups or SIMD-group functions, which CoreAnimation asks
 * for explicitly and then avoids.
 */
#import "s8mtl.h"

size_t strlen(const char *);
void *memset(void *, int, size_t);
int snprintf(char *, size_t, const char *, ...);
int strncmp(const char *, const char *, size_t);
char *strstr(const char *, const char *);
int sysctlbyname(const char *, void *, size_t *, void *, size_t);
const char *getprogname(void);

/* 0 with boot-arg s8vtcompute=0: VideoToolbox's GPU paths are refused
 * (newComputePipelineStateWithFunction:error:, render pipelines below). */
static int s8_vt_gpu(void)
{
	static int on = -1;
	if (on < 0) {
		char args[1024] = "";
		size_t alen = sizeof(args) - 1;
		on = !(sysctlbyname("kern.bootargs", args, &alen, NULL, 0) == 0 && strstr(args, "s8vtcompute=0") != NULL);
	}
	return on;
}

@implementation S8MaliDevice

- (instancetype)initWithAcceleratorPort:(unsigned int)port
{
	self = [super init];
	if (self == nil)
		return nil;
	_s8port = port;
	struct s8mali_info info = { 0 };
	int ioctl(int, unsigned long, ...);
	if (s8_fd() >= 0)
		ioctl(s8_fd(), S8MALI_INFO, &info);
	s8_log("MTL device created (stage 3): gpu 0x%x ready %u, %llu MiB GPU memory free, draw path %s",
	    info.gpu_id, info.ready, (unsigned long long)(info.heap_free >> 20), s8d_init() == 0 ? "on" : "off");
	return self;
}

S8_FORWARDING

- (id)name
{
	return [NSString stringWithUTF8String:"Mali-G71 MP20 (Galaxy S8)"];
}

- (NSUInteger)featureProfile { return 5; }
- (NSUInteger)maxBufferLength { return 256u << 20; }
- (BOOL)supportsTileShaders { return NO; }
- (BOOL)supportsMemorylessRenderTargets { return NO; }
- (BOOL)supportsRasterOrderGroups { return NO; }
- (BOOL)supportsSIMDGroup { return NO; }
- (BOOL)supportsExtendedXR10Formats { return NO; }
- (BOOL)supportsYCBCRFormats12 { return NO; }
- (BOOL)supportsYCBCRPackedFormats12 { return NO; }
- (BOOL)supportsNativeHardwareFP16 { return YES; }
- (BOOL)isLowPower { return YES; }
- (BOOL)isHeadless { return NO; }
- (BOOL)isRemovable { return NO; }
- (BOOL)hasUnifiedMemory { return YES; }
- (BOOL)supportsVertexAmplificationCount:(NSUInteger)count { return count <= 1; }

/* ---- queues ---- */

- (id)newCommandQueue
{
	return [[S8CommandQueue alloc] initWithS8Device:self];
}

- (id)newCommandQueueWithMaxCommandBufferCount:(NSUInteger)count
{
	(void)count;
	return [[S8CommandQueue alloc] initWithS8Device:self];
}

- (id)newCommandQueueWithDescriptor:(id)descriptor
{
	(void)descriptor;
	return [[S8CommandQueue alloc] initWithS8Device:self];
}

/* ---- resources ---- */

- (id)newBufferWithLength:(NSUInteger)length options:(NSUInteger)options
{
	return [[S8Buffer alloc] initWithDevice:self length:length options:options];
}

- (id)newBufferWithBytes:(const void *)bytes length:(NSUInteger)length options:(NSUInteger)options
{
	void *memcpy(void *, const void *, size_t);
	S8Buffer *b = [[S8Buffer alloc] initWithDevice:self length:length options:options];
	if (b != nil && bytes != NULL)
		memcpy(b->_mem.cpu, bytes, length);
	return b;
}

- (id)newBufferWithBytesNoCopy:(void *)p length:(NSUInteger)length options:(NSUInteger)options
    deallocator:(id)dealloc
{
	return [[S8Buffer alloc] initWithDevice:self bytesNoCopy:p length:length options:options
	    deallocator:dealloc];
}

- (id)newTextureWithDescriptor:(id)desc
{
	return [[S8Texture alloc] initWithDevice:self descriptor:desc];
}

- (id)newTextureWithDescriptor:(id)desc iosurface:(void *)surface plane:(NSUInteger)plane
{
	return [[S8Texture alloc] initWithDevice:self descriptor:desc iosurface:surface plane:plane];
}

- (id)newSamplerStateWithDescriptor:(id)desc
{
	return [[S8SamplerState alloc] initWithDevice:self samplerDescriptor:desc];
}

- (id)newDepthStencilStateWithDescriptor:(id)desc
{
	return [[S8DepthStencilState alloc] initWithDevice:self depthStencilDescriptor:desc];
}

- (id)newHeapWithDescriptor:(id)desc
{
	static unsigned logged;
	S8Heap *h = [[S8Heap alloc] initWithDevice:self descriptor:desc];
	if (logged++ < 8)
		s8_log("MTL heap: %lu KiB, storage %lu", (unsigned long)(h ? h->_size >> 10 : 0),
		    (unsigned long)(h ? h->_storage : 0));
	return h;
}

/* IORegistry entry ID of the accelerator (stable per boot). */
- (uint64_t)registryID { return 0x100000000ULL | _s8port; }

/* ---- shaders and pipelines ---- */

- (id)newLibraryWithURL:(id)url error:(id *)error
{
	if (error != NULL)
		*error = nil;
	return [[S8Library alloc] initWithDevice:self path:[[url path] UTF8String]];
}

/*
 * The other ways to get a library. Unimplemented they went to Apple's
 * _MTLDevice, whose library could not make the functions without Apple's
 * compiler service: MetalPerformanceShaders in SpringBoard failed with
 * "MTLComputePipelineStateCache unable to load function" and aborted, so the
 * lock screen never drew. Kernels get stub compute pipelines (dispatches are
 * skipped, newComputePipelineStateWithFunction).
 */
- (id)newLibraryWithFile:(id)path error:(id *)error
{
	if (error != NULL)
		*error = nil;
	return [[S8Library alloc] initWithDevice:self path:[path UTF8String]];
}

- (id)newDefaultLibraryWithBundle:(id)bundle error:(id *)error
{
	char p[1024];
	if (error != NULL)
		*error = nil;
	const char *dir = bundle ? [[bundle bundlePath] UTF8String] : NULL;
	snprintf(p, sizeof(p), "%s/default.metallib", dir ? dir : ".");
	return [[S8Library alloc] initWithDevice:self path:p];
}

- (id)newDefaultLibrary
{
	return [self newDefaultLibraryWithBundle:[objc_getClass("NSBundle") mainBundle] error:NULL];
}

void *dispatch_data_create_map(void *data, const void **buffer, size_t *size);

- (id)newLibraryWithData:(id)data error:(id *)error
{
	const void *bytes = NULL;
	size_t len = 0;
	if (error != NULL)
		*error = nil;
	id map = data ? (id)dispatch_data_create_map((void *)data, &bytes, &len) : nil;
	char what[48];
	snprintf(what, sizeof(what), "(data, %zu bytes)", len);
	id lib = [[S8Library alloc] initWithDevice:self bytes:bytes length:len what:what];
	[map release];
	return lib;
}

static id s8_error(const char *what)
{
	return [NSError errorWithDomain:[NSString stringWithUTF8String:what] code:1 userInfo:nil];
}

- (id)newPipelineLibraryWithFilePath:(id)path error:(id *)error
{
	(void)path;
	if (error != NULL)
		*error = s8_error("S8MaliMetal: no precompiled pipelines for the Mali");
	return nil;
}

- (id)newRenderPipelineStateWithDescriptor:(id)desc error:(id *)error
{
	return [self newRenderPipelineStateWithDescriptor:desc options:0 reflection:NULL error:error];
}

- (id)newRenderPipelineStateWithDescriptor:(id)desc options:(NSUInteger)options
    reflection:(id *)reflection error:(id *)error
{
	static unsigned count;
	(void)options;
	if (reflection != NULL)
		*reflection = nil;
	if (error != NULL)
		*error = nil;
	S8RenderPipelineState *p = [[S8RenderPipelineState alloc] init];
	p->_s8device = self;
	p->_vertex = [[desc vertexFunction] retain];
	p->_fragment = [[desc fragmentFunction] retain];
	id ca = [[desc colorAttachments] objectAtIndexedSubscript:0];
	p->_colorFormat = [ca pixelFormat];
	p->_blending = [ca isBlendingEnabled];
	p->_id = ++count;
	S8Function *vs = p->_vertex, *fs = p->_fragment;
	/* Functions from other libraries (SpringBoard's own shaders, compiled
	 * by MTLCompilerService, boot 51) are Metal's objects, not ours: no
	 * GPU pipeline for them (their draws are skipped). */
	BOOL foreign = (vs != nil && ![vs isKindOfClass:[S8Function class]]) ||
	    (fs != nil && ![fs isKindOfClass:[S8Function class]]);
	if (foreign) {
		static unsigned logged;
		if (logged++ < 20)
			s8_log("MTL pipeline %u: foreign functions %s %s / %s %s (no GPU pipeline)", p->_id,
			    vs ? class_getName(object_getClass(vs)) : "-", vs ? [[(id)vs name] UTF8String] : "",
			    fs ? class_getName(object_getClass(fs)) : "-", fs ? [[(id)fs name] UTF8String] : "");
		return p;
	}
	s8_log("MTL pipeline %u: vs %s fs %s fmt %lu blend %d", p->_id,
	    vs ? [vs->_name UTF8String] : "-", fs ? [fs->_name UTF8String] : "-",
	    (unsigned long)p->_colorFormat, p->_blending);
	/*
	 * The exact constant sets, once per process each, for build-shaders.py
	 * --spec (test/fc2spec.py reads them back). Pieces fit the 119
	 * characters of a log line: "MTL fc <16 hex>+<off>: " and 80 of the key
	 * (the 96-character pieces of boot mps1 lost their last 8).
	 */
	for (int which = 0; which < 2; which++) {
		static uint64_t seen[256];
		static unsigned nseen;
		S8Function *fn = which ? fs : vs;
		if (fn == nil || !fn->_constants[0])
			continue;
		size_t len = strlen(fn->_constants);
		uint64_t h = s8d_key_hash(fn->_constants);
		unsigned k;
		for (k = 0; k < nseen && seen[k] != h; k++)
			;
		if (k < nseen)
			continue;
		if (nseen < 256)
			seen[nseen++] = h;
		s8_log("MTL fc %016llx: pipeline %u %s %.40s, %zu chars%s", (unsigned long long)h, p->_id,
		    which ? "fs" : "vs", [fn->_name UTF8String], len,
		    len + 1 >= sizeof(fn->_constants) ? " TRUNCATED" : "");
		for (size_t o = 0; o < len; o += 80)
			s8_log("MTL fc %016llx+%zu: %.80s", (unsigned long long)h, o, fn->_constants + o);
	}
	/* GPU pipeline (stage 3) */
	struct s8d_pipeline_desc d;
	memset(&d, 0, sizeof(d));
	d.vs_name = vs ? [vs->_name UTF8String] : NULL;
	d.fs_name = fs ? [fs->_name UTF8String] : NULL;
	d.vs_constants = vs ? vs->_constants : "";
	d.fs_constants = fs ? fs->_constants : "";
	d.color_format = (uint32_t)p->_colorFormat;
	d.blending = p->_blending;
	d.write_mask = (uint8_t)[ca writeMask];
	d.rgb_op = (uint8_t)[ca rgbBlendOperation];
	d.alpha_op = (uint8_t)[ca alphaBlendOperation];
	d.src_rgb = (uint8_t)[ca sourceRGBBlendFactor];
	d.dst_rgb = (uint8_t)[ca destinationRGBBlendFactor];
	d.src_alpha = (uint8_t)[ca sourceAlphaBlendFactor];
	d.dst_alpha = (uint8_t)[ca destinationAlphaBlendFactor];
	id vd = [desc vertexDescriptor];
	if (vd != nil) {
		id attrs = [vd attributes], layouts = [vd layouts];
		for (unsigned i = 0; i < S8D_MAX_ATTRIBS; i++) {
			id a = [attrs objectAtIndexedSubscript:i];
			d.attribs[i].format = (uint32_t)[a format];
			d.attribs[i].offset = (uint32_t)[a offset];
			d.attribs[i].buffer = (uint32_t)[a bufferIndex];
		}
		for (unsigned i = 0; i < S8D_MAX_BUFFERS; i++) {
			id l = [layouts objectAtIndexedSubscript:i];
			d.layouts[i].stride = (uint32_t)[l stride];
			d.layouts[i].step = (uint32_t)[l stepFunction];
			d.layouts[i].rate = (uint32_t)[l stepRate];
		}
	}
	p->_gpu = s8d_pipeline_new(&d, p->_id);
	/* s8vtcompute=0: VideoToolbox's render path cannot run either (its
	 * fragment shaders need local arrays air2nir does not translate yet),
	 * so refuse it too and leave VideoToolbox the CPU. */
	if (p->_gpu == NULL && !s8_vt_gpu() && fs != nil && strncmp([fs->_name UTF8String], "VTMTS", 5) == 0) {
		s8_log("MTL pipeline %u: VideoToolbox render pipeline refused (s8vtcompute=0) [%s]", p->_id,
		    getprogname());
		[p release];
		if (error != NULL)
			*error = [objc_getClass("NSError") errorWithDomain:[objc_getClass("NSString")
			    stringWithUTF8String:"S8MaliMetal"] code:2 userInfo:nil];
		return nil;
	}
	return p;
}

- (id)newComputePipelineStateWithFunction:(id)function error:(id *)error
{
	static unsigned logged;
	if (error != NULL)
		*error = nil;
	const char *name = function ? [[function name] UTF8String] : NULL;
	if (name == NULL)
		name = "-";
	/*
	 * Compute kernels do not run yet (dispatches are skipped), so whatever
	 * a kernel should write stays zero. VideoToolbox's pixel transfers
	 * (VTMTSCompute*) made SpringBoard's wallpaper image that way: black
	 * (boot f16). Boot-arg s8vtcompute=0 refuses those pipelines with an
	 * error, so VideoToolbox can take its render-pipeline path
	 * (VTMTSVertexFunction/VTMTSFragmentFunction, in the shader cache) or
	 * the CPU instead.
	 */
	if (!s8_vt_gpu() && strncmp(name, "VTMTSCompute", 12) == 0) {
		s8_log("MTL compute pipeline %.40s refused (s8vtcompute=0) [%s]", name, getprogname());
		if (error != NULL)
			*error = [objc_getClass("NSError") errorWithDomain:[objc_getClass("NSString")
			    stringWithUTF8String:"S8MaliMetal"] code:1 userInfo:nil];
		return nil;
	}
	if (logged++ < 40)
		s8_log("MTL compute pipeline %.40s (dispatches skipped) [%s]", name, getprogname());
	S8ComputePipelineState *p = [[S8ComputePipelineState alloc] init];
	p->_s8device = self;
	snprintf(p->_name, sizeof(p->_name), "%s", name);
	return p;
}

/* The other compute pipeline requests (MetalPerformanceShaders), all to the
 * same stub; reflection is not provided. */
- (id)newComputePipelineStateWithFunction:(id)function options:(NSUInteger)options
    reflection:(id *)reflection error:(id *)error
{
	(void)options;
	if (reflection != NULL)
		*reflection = nil;
	return [self newComputePipelineStateWithFunction:function error:error];
}

- (id)newComputePipelineStateWithDescriptor:(id)desc options:(NSUInteger)options
    reflection:(id *)reflection error:(id *)error
{
	(void)options;
	if (reflection != NULL)
		*reflection = nil;
	return [self newComputePipelineStateWithFunction:[desc computeFunction] error:error];
}

/* Blocks (Apple ABI): isa, flags, reserved, invoke(block, ...) */
struct s8_handler_block {
	void *isa;
	int flags, reserved;
	void (*invoke)(void *block, ...);
};

- (void)newComputePipelineStateWithFunction:(id)function completionHandler:(id)handler
{
	id p = [self newComputePipelineStateWithFunction:function error:NULL];
	if (handler != nil)
		((struct s8_handler_block *)handler)->invoke(handler, p, nil);
	[p release];
}

- (void)newComputePipelineStateWithFunction:(id)function options:(NSUInteger)options
    completionHandler:(id)handler
{
	(void)options;
	id p = [self newComputePipelineStateWithFunction:function error:NULL];
	if (handler != nil)
		((struct s8_handler_block *)handler)->invoke(handler, p, nil, nil);
	[p release];
}

- (void)newComputePipelineStateWithDescriptor:(id)desc options:(NSUInteger)options
    completionHandler:(id)handler
{
	(void)options;
	id p = [self newComputePipelineStateWithFunction:[desc computeFunction] error:NULL];
	if (handler != nil)
		((struct s8_handler_block *)handler)->invoke(handler, p, nil, nil);
	[p release];
}

- (id)newRenderPipelineStateWithTileDescriptor:(id)desc options:(NSUInteger)options
    reflection:(id *)reflection error:(id *)error
{
	(void)desc;
	(void)options;
	if (reflection != NULL)
		*reflection = nil;
	if (error != NULL)
		*error = s8_error("S8MaliMetal: no tile shaders");
	return nil;
}

@end
