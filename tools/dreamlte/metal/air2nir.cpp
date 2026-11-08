/*
 * air2nir: Apple Metal AIR bitcode -> Mesa NIR -> Mali Bifrost (v6) binary.
 *
 * Host tool of the Galaxy S8 Metal driver (S8MaliMetal). CoreAnimation's
 * shaders ship as AIR (LLVM IR, triple air64-apple-ios) in QuartzCore's
 * default.metallib and are specialised at run time with Metal function
 * constants. This tool does what Apple's back end does for the AGX GPU,
 * for the Mali-G71 instead:
 *
 *  1. read one function's bitcode from a metallib (LLVM 18 reads AIR);
 *  2. specialise: function-constant globals (MTL_FC_INIT_n) become their
 *     values, air.is_function_constant_defined folds, the module's
 *     air.static_init function (which derives the implicit
 *     __metal_implicit_fc_pred_* globals) is evaluated by GlobalOpt, then
 *     everything is inlined and optimised (-O2, fast math);
 *  3. translate the LLVM IR to NIR, unstructured (one NIR block per LLVM
 *     block, gotos, phis through local variables, as spirv_to_nir does for
 *     OpenCL), then nir_lower_goto_ifs; Metal's interface (stage_in
 *     attributes, varyings by name, render targets, buffers, textures,
 *     samplers) becomes NIR variables, UBOs and texture instructions;
 *  4. compile with Mesa's Bifrost compiler (pan_shader_preprocess /
 *     pan_shader_compile) and write the binary plus the interface the
 *     driver needs (see the output format at the end of this file).
 *
 * Driver ABI (shared with s8mtl_draw.c):
 *  - Metal buffer index N in address space 2 (constant) -> UBO N;
 *    device buffers (address space 1) -> UBO S8_UBO_ADDRS holds their
 *    64-bit GPU addresses, loads are global;
 *  - texture index N -> texture N; sampler index N -> sampler N; inline
 *    constexpr samplers -> sampler 16 + k (states listed in the output);
 *  - vertex attribute (stage_in) location L -> attribute L;
 *  - varyings: matched by Metal's generated name; slot VAR0 + k where k is
 *    the name's index in the varying table passed with -v (same table for
 *    all shaders of a build);
 *  - render target i -> FRAG_RESULT_DATA0 + i;
 *  - UBO S8_UBO_SYSVALS: viewport scale at 0 and offset at 16 (vec3 each),
 *    render target conversions at 32 + (rt * 2 + is16) * 4, the draw's
 *    first vertex at 96, texture sizes at 128 + index * 8.
 */
#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/ReplaceConstant.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/IPO/GlobalOpt.h>
#include <llvm/Transforms/Scalar/EarlyCSE.h>
#include <llvm/Transforms/Scalar/SROA.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_builtin_builder.h"
#include "compiler/glsl_types.h"
#include "util/u_dynarray.h"
/* Panfrost's headers have no extern "C" of their own. */
extern "C" {
#include "pan_shader.h"
}

using namespace llvm;

#define S8_UBO_DATA    28      /* constant tables of the shader */
#define S8_UBO_FC      29
#define S8_UBO_ADDRS   30
#define S8_UBO_SYSVALS 31
#define S8_INLINE_SAMPLER_BASE 16
/* UBO S8_UBO_SYSVALS layout (shared with s8mtl_draw.c) */
#define S8_SV_VERTEX_OFFSET 96         /* u32 */
#define S8_SV_TEX_SIZE 128             /* 32 x { u32 width, height } */
#define S8_SV_SIZE (128 + 32 * 8)
#define S8_NULL_TEXTURE 31        /* bound to a 1x1 transparent black texture */

static void fail(const std::string &msg)
{
	fprintf(stderr, "air2nir: %s\n", msg.c_str());
	exit(1);
}

/* nir_load_ubo with alignment and an unbounded range (C++ cannot mix the
 * builder macros' designated initialisers with positional ones). */
static nir_def *s8_load_ubo(nir_builder *b, unsigned nc, unsigned bs, nir_def *index, nir_def *offset,
    unsigned align)
{
	nir_def *d = nir_load_ubo(b, nc, bs, index, offset);
	nir_intrinsic_instr *in = nir_instr_as_intrinsic(d->parent_instr);
	nir_intrinsic_set_align(in, align, 0);
	nir_intrinsic_set_range_base(in, 0);
	nir_intrinsic_set_range(in, ~0u);
	return d;
}

/* ---- metallib ---- */

struct MetallibFunction {
	std::string name;
	unsigned type;
	std::vector<uint8_t> bitcode;
};

static std::vector<MetallibFunction> read_metallib(const std::string &path)
{
	std::ifstream f(path, std::ios::binary);
	std::vector<uint8_t> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
	if (d.size() < 0x58 || memcmp(d.data(), "MTLB", 4) != 0)
		fail(path + ": not a metallib");
	auto u64 = [&](size_t o) { uint64_t v; memcpy(&v, &d[o], 8); return v; };
	auto u32 = [&](size_t o) { uint32_t v; memcpy(&v, &d[o], 4); return v; };
	auto u16 = [&](size_t o) { uint16_t v; memcpy(&v, &d[o], 2); return v; };
	uint64_t list = u64(0x18), bc = u64(0x48);
	uint32_t count = u32(list);
	std::vector<MetallibFunction> out;
	size_t off = list + 4;
	for (uint32_t i = 0; i < count; i++) {
		uint32_t len = u32(off);
		size_t p = off + 4, end = off + len;
		MetallibFunction fn;
		uint64_t size = 0, bcoff = 0;
		while (p + 6 <= end && memcmp(&d[p], "ENDT", 4) != 0) {
			uint16_t sz = u16(p + 4);
			const uint8_t *v = &d[p + 6];
			if (!memcmp(&d[p], "NAME", 4))
				fn.name.assign((const char *)v, strnlen((const char *)v, sz));
			else if (!memcmp(&d[p], "TYPE", 4))
				fn.type = v[0];
			else if (!memcmp(&d[p], "MDSZ", 4))
				memcpy(&size, v, 8);
			else if (!memcmp(&d[p], "OFFT", 4))
				memcpy(&bcoff, v + 16, 8);
			p += 6 + sz;
		}
		fn.bitcode.assign(d.begin() + bc + bcoff, d.begin() + bc + bcoff + size);
		out.push_back(fn);
		off = end;
	}
	return out;
}

/* ---- specialisation ---- */

struct ConstantValue {
	std::string name;
	uint64_t bits;
};

/* A function constant read at run time: its value and "defined" flag
 * live in UBO S8_UBO_FC at index * 16 (value, little endian) and
 * index * 16 + 8 (u32 non-zero when defined). */
struct DynamicConstant {
	std::string name;
	unsigned index, bits;
};

static std::vector<DynamicConstant> dynamic_constants;

static void make_dynamic(Module &m, Function &entry)
{
	LLVMContext &ctx = m.getContext();
	NamedMDNode *fcs = m.getNamedMetadata("air.function_constants");
	if (fcs) {
		for (MDNode *n : fcs->operands()) {
			auto *gv = mdconst::dyn_extract_or_null<GlobalVariable>(n->getOperand(0));
			auto *name = dyn_cast<MDString>(n->getOperand(2));
			auto *idx = mdconst::dyn_extract_or_null<ConstantInt>(n->getOperand(3));
			if (!gv || !name || !idx)
				continue;
			unsigned index = (unsigned)idx->getZExtValue();
			Type *t = gv->getValueType();
			dynamic_constants.push_back({ name->getString().str(), index,
			    (unsigned)m.getDataLayout().getTypeStoreSizeInBits(t) });
			std::vector<User *> users(gv->user_begin(), gv->user_end());
			for (User *u : users) {
				auto *ins = dyn_cast<Instruction>(u);
				if (!ins)
					fail("function constant " + name->getString().str() + " used by a constant expression");
				IRBuilder<> b(ins);
				if (auto *ci = dyn_cast<CallInst>(ins)) {
					FunctionCallee fc = m.getOrInsertFunction("s8.fcdef", Type::getInt1Ty(ctx), b.getInt32Ty());
					ci->replaceAllUsesWith(b.CreateCall(fc, { b.getInt32(index) }));
					ci->eraseFromParent();
				} else if (auto *ld = dyn_cast<LoadInst>(ins)) {
					std::string tn;
					raw_string_ostream os(tn);
					ld->getType()->print(os);
					FunctionCallee fc = m.getOrInsertFunction("s8.fc." + os.str(), ld->getType(), b.getInt32Ty());
					ld->replaceAllUsesWith(b.CreateCall(fc, { b.getInt32(index) }));
					ld->eraseFromParent();
				} else {
					fail("function constant " + name->getString().str() + " has an unsupported user");
				}
			}
		}
	}
	/* The static initialiser now runs at the start of the entry point. */
	for (Function &f : m) {
		if (f.getSection() != "air.static_init")
			continue;
		f.setSection("");
		f.setLinkage(GlobalValue::InternalLinkage);
		IRBuilder<> b(&*entry.getEntryBlock().getFirstInsertionPt());
		CallInst *call = b.CreateCall(&f);
		call->setCallingConv(f.getCallingConv());
		InlineFunctionInfo ifi;
		InlineResult ir = InlineFunction(*call, ifi);
		if (!ir.isSuccess())
			fail(std::string("cannot inline the static initialiser: ") + ir.getFailureReason());
	}
	/* ... and nothing else: llvm.global_ctors kept the initialiser's own
	 * body alive, and localise_globals then repointed its stores at the
	 * entry point's allocas across functions (invalid IR the O2 pipeline
	 * happened to survive, and crashed on with -F). */
	if (GlobalVariable *ctors = m.getGlobalVariable("llvm.global_ctors"))
		ctors->eraseFromParent();
}

/* Point the users of old (a global's address or a GEP of it) at new, a
 * private alloca in address space 0. */
static void repoint(Value *old, Value *nw);

/* v selects between a localised global (now nw_side, address space 0) and
 * another pointer: push the choice down to the loads. a and b are the
 * true/false side pointers after the same GEPs as v. */
static void distribute(Value *v, Value *cond, Value *a, Value *b)
{
	std::vector<User *> users(v->user_begin(), v->user_end());
	for (User *u : users) {
		auto *ins = dyn_cast<Instruction>(u);
		if (!ins)
			fail("unsupported constant use of a selected shader global");
		IRBuilder<> bl(ins);
		if (auto *ld = dyn_cast<LoadInst>(ins)) {
			LoadInst *la = bl.CreateLoad(ld->getType(), a);
			LoadInst *lb = bl.CreateLoad(ld->getType(), b);
			la->setAlignment(ld->getAlign());
			lb->setAlignment(ld->getAlign());
			ld->replaceAllUsesWith(bl.CreateSelect(cond, la, lb));
		} else if (auto *g = dyn_cast<GetElementPtrInst>(ins)) {
			std::vector<Value *> idx(g->idx_begin(), g->idx_end());
			distribute(g, cond, bl.CreateGEP(g->getSourceElementType(), a, idx, "", g->isInBounds()),
			    bl.CreateGEP(g->getSourceElementType(), b, idx, "", g->isInBounds()));
		} else {
			std::string s;
			raw_string_ostream os(s);
			ins->print(os);
			fail("unsupported use of a selected shader global: " + os.str());
		}
		ins->eraseFromParent();
	}
}

static void repoint(Value *old, Value *nw)
{
	std::vector<User *> users(old->user_begin(), old->user_end());
	for (User *u : users) {
		auto *ins = dyn_cast<Instruction>(u);
		if (!ins)
			fail("unsupported constant use of a shader global");
		IRBuilder<> b(ins);
		if (auto *ld = dyn_cast<LoadInst>(ins)) {
			LoadInst *n = b.CreateLoad(ld->getType(), nw);
			n->setAlignment(ld->getAlign());
			ld->replaceAllUsesWith(n);
		} else if (auto *st = dyn_cast<StoreInst>(ins)) {
			if (st->getPointerOperand() != old)
				fail("shader global address stored as a value");
			b.CreateAlignedStore(st->getValueOperand(), nw, st->getAlign());
		} else if (auto *g = dyn_cast<GetElementPtrInst>(ins)) {
			std::vector<Value *> idx(g->idx_begin(), g->idx_end());
			repoint(g, b.CreateGEP(g->getSourceElementType(), nw, idx, "", g->isInBounds()));
		} else if (auto *c = dyn_cast<AddrSpaceCastInst>(ins)) {
			repoint(c, nw);
		} else if (auto *sel = dyn_cast<SelectInst>(ins); sel && sel->getCondition() != old) {
			bool t = sel->getTrueValue() == old;
			Value *other = t ? sel->getFalseValue() : sel->getTrueValue();
			distribute(sel, sel->getCondition(), t ? nw : other, t ? other : nw);
		} else if (auto *ci = dyn_cast<CallInst>(ins); ci && ci->getCalledFunction() &&
		    ci->getCalledFunction()->isIntrinsic() &&
		    ci->getCalledFunction()->getName().starts_with("llvm.lifetime")) {
		} else {
			std::string s;
			raw_string_ostream os(s);
			ins->print(os);
			fail("unsupported use of a shader global: " + os.str());
		}
		ins->eraseFromParent();
	}
}

/* After inlining: globals the shader writes (the implicit predicates, the
 * fragment shader configuration) become private variables of the entry
 * point. */
static void localise_globals(Module &m, Function &entry)
{
	for (const char *keep : { "llvm.used", "llvm.compiler.used" })
		if (GlobalVariable *g = m.getGlobalVariable(keep))
			g->eraseFromParent();
	/* The inlined functions' own bodies are dead now; they would keep
	 * using the globals, and repointing those uses at the entry point's
	 * allocas made invalid IR (instructions of one function using
	 * another's values: the verifier's "does not dominate all uses").
	 * The O2 pipeline used to drop them before anything noticed; with
	 * folded constants (-F) it crashed instead. */
	for (bool changed = true; changed;) {
		changed = false;
		for (Function &f : make_early_inc_range(m)) {
			if (&f == &entry || f.isDeclaration())
				continue;
			f.removeDeadConstantUsers();
			if (f.use_empty()) {
				f.eraseFromParent();
				changed = true;
			}
		}
	}
	std::vector<GlobalVariable *> gvs;
	for (GlobalVariable &gv : m.globals())
		if (!gv.isConstant() && gv.hasLocalLinkage())
			gvs.push_back(&gv);
	for (GlobalVariable *gv : gvs) {
		gv->removeDeadConstantUsers();
		if (gv->use_empty())
			continue;
		convertUsersOfConstantsToInstructions({ gv });
		IRBuilder<> eb(&*entry.getEntryBlock().getFirstInsertionPt());
		AllocaInst *a = eb.CreateAlloca(gv->getValueType());
		a->setAlignment(gv->getAlign().valueOrOne());
		if (gv->hasInitializer())
			eb.CreateStore(gv->getInitializer(), a);
		repoint(gv, a);
	}
}

/*
 * -D -F: the run-time constant reads of -D (s8.fcdef / s8.fc.*) become the
 * values a key gives, exactly as the driver would fill UBO S8_UBO_FC from
 * it (s8_fc_ubo: the value's 8 bytes little endian, "defined" when the key
 * names the constant, zero otherwise). A specialisation is then the same
 * program as the generic shader running with that key, folded.
 */
static bool fold_mode;
static std::map<std::string, uint64_t> fold_values;

static Constant *fold_value(Type *t, uint64_t bits)
{
	auto *vt = dyn_cast<FixedVectorType>(t);
	Type *et = vt ? vt->getElementType() : t;
	unsigned n = vt ? vt->getNumElements() : 1;
	unsigned eb = et->getPrimitiveSizeInBits(), eb_store = eb == 1 ? 8 : eb;
	std::vector<Constant *> elems;
	for (unsigned i = 0; i < n; i++) {
		unsigned shift = i * eb_store;
		uint64_t v = shift < 64 ? bits >> shift : 0;
		if (eb_store < 64)
			v &= (1ull << eb_store) - 1;
		Constant *c;
		if (et->isIntegerTy())
			c = ConstantInt::get(et, eb == 1 ? (v != 0) : v);
		else if (et->isHalfTy() || et->isFloatTy() || et->isDoubleTy())
			c = ConstantFP::get(t->getContext(), APFloat(et->isHalfTy() ? APFloat::IEEEhalf() :
			    et->isFloatTy() ? APFloat::IEEEsingle() : APFloat::IEEEdouble(), APInt(eb, v)));
		else
			fail("folded function constant of unsupported type");
		elems.push_back(c);
	}
	return vt ? ConstantVector::get(elems) : elems[0];
}

static void fold_dynamic(Module &m)
{
	std::map<unsigned, std::string> names;
	for (auto &dc : dynamic_constants)
		names[dc.index] = dc.name;
	/* "@index" entries (set by index in Metal) name the constant by its
	 * index; they come after the names and win, as in the driver */
	std::map<std::string, uint64_t> by_index;
	for (auto it = fold_values.begin(); it != fold_values.end();) {
		if (it->first.size() > 1 && it->first[0] == '@') {
			unsigned index = (unsigned)strtoul(it->first.c_str() + 1, nullptr, 10);
			if (names.count(index))
				by_index[names[index]] = it->second;
			else
				fprintf(stderr, "air2nir: note: constant index %u is not the function's\n", index);
			it = fold_values.erase(it);
		} else {
			++it;
		}
	}
	for (auto &[name, v] : by_index)
		fold_values[name] = v;
	std::vector<CallInst *> calls;
	for (Function &f : m)
		if (f.getName() == "s8.fcdef" || f.getName().starts_with("s8.fc."))
			for (User *u : f.users())
				if (auto *ci = dyn_cast<CallInst>(u))
					calls.push_back(ci);
	for (CallInst *ci : calls) {
		unsigned index = (unsigned)cast<ConstantInt>(ci->getArgOperand(0))->getZExtValue();
		auto it = names.find(index);
		bool defined = it != names.end() && fold_values.count(it->second) != 0;
		uint64_t bits = defined ? fold_values[it->second] : 0;
		Constant *c = ci->getCalledFunction()->getName() == "s8.fcdef" ?
		    ConstantInt::getBool(m.getContext(), defined) : fold_value(ci->getType(), bits);
		ci->replaceAllUsesWith(c);
		ci->eraseFromParent();
	}
	for (auto &[name, v] : fold_values) {
		bool known = false;
		for (auto &dc : dynamic_constants)
			known |= dc.name == name;
		if (!known)
			fprintf(stderr, "air2nir: note: constant %s is not the function's\n", name.c_str());
	}
	dynamic_constants.clear();      /* nothing left to read at run time */
}

/*
 * Texture binding indices given by function-constant expressions
 * ([[texture(expr)]]: the argument's air.location_index operand is a global
 * that the static initialiser computes; VideoToolbox's shaders). Every use
 * of such an argument becomes s8.texidx(load global): textures are their
 * binding index for the translator, so the index is a constant once the
 * constants are folded and a run-time (texture_offset) index in -D shaders.
 * Done before anything else, while the arguments' metadata still names the
 * globals.
 */
static void lower_expr_locations(Module &m, Function &entry)
{
	LLVMContext &ctx = m.getContext();
	std::vector<std::pair<unsigned, GlobalVariable *>> found;
	for (const char *kind : { "air.vertex", "air.fragment", "air.kernel" }) {
		NamedMDNode *nm = m.getNamedMetadata(kind);
		if (!nm)
			continue;
		for (MDNode *n : nm->operands()) {
			if (n->getNumOperands() == 0 || mdconst::dyn_extract_or_null<Function>(n->getOperand(0)) != &entry)
				continue;
			for (const MDOperand &lop : n->operands()) {
				auto *list = dyn_cast_or_null<MDNode>(lop.get());
				if (!list)
					continue;
				for (const MDOperand &aop : list->operands()) {
					auto *a = dyn_cast_or_null<MDNode>(aop.get());
					if (!a || a->getNumOperands() < 2)
						continue;
					auto *argno = mdconst::dyn_extract_or_null<ConstantInt>(a->getOperand(0));
					if (!argno)
						continue;
					bool texture = false;
					GlobalVariable *gv = nullptr;
					for (unsigned i = 1; i + 1 < a->getNumOperands(); i++) {
						auto *s = dyn_cast_or_null<MDString>(a->getOperand(i).get());
						if (s && s->getString() == "air.texture")
							texture = true;
						if (s && s->getString() == "air.location_index")
							gv = mdconst::dyn_extract_or_null<GlobalVariable>(a->getOperand(i + 1));
					}
					if (gv && texture)
						found.push_back({ (unsigned)argno->getZExtValue(), gv });
					else if (gv)
						fail("binding index by function-constant expression for a non-texture argument");
				}
			}
		}
	}
	for (auto &[argno, gv] : found) {
		if (argno >= entry.arg_size())
			fail("argument index out of range in metadata");
		Argument *arg = entry.getArg(argno);
		FunctionCallee fc = m.getOrInsertFunction("s8.texidx", arg->getType(), Type::getInt32Ty(ctx));
		std::vector<Use *> uses;
		for (Use &u : arg->uses())
			uses.push_back(&u);
		for (Use *u : uses) {
			auto *ins = cast<Instruction>(u->getUser());
			/* before the use (in a phi's incoming block: before its end) */
			Instruction *at = ins;
			if (auto *phi = dyn_cast<PHINode>(ins))
				at = phi->getIncomingBlock(*u)->getTerminator();
			IRBuilder<> b(at);
			Value *loc = b.CreateLoad(Type::getInt32Ty(ctx), gv);
			u->set(b.CreateCall(fc, { loc }));
		}
	}
}

static void specialise(Module &m, Function &entry, const std::vector<ConstantValue> &values, bool dynamic)
{
	lower_expr_locations(m, entry);
	if (dynamic) {
		make_dynamic(m, entry);
		for (Function &f : m) {
			if (&f == &entry || f.isDeclaration())
				continue;
			f.removeFnAttr(Attribute::NoInline);
			f.addFnAttr(Attribute::AlwaysInline);
		}
		{
			LoopAnalysisManager lam;
			FunctionAnalysisManager fam;
			CGSCCAnalysisManager cam;
			ModuleAnalysisManager mam;
			PassBuilder pb;
			pb.registerModuleAnalyses(mam);
			pb.registerCGSCCAnalyses(cam);
			pb.registerFunctionAnalyses(fam);
			pb.registerLoopAnalyses(lam);
			pb.crossRegisterProxies(lam, fam, cam, mam);
			ModulePassManager mpm;
			mpm.addPass(AlwaysInlinerPass());
			mpm.run(m, mam);
		}
		{
			FunctionAnalysisManager fam;
			PassBuilder pb;
			pb.registerFunctionAnalyses(fam);
			FunctionPassManager fpm;
			fpm.addPass(SROAPass(SROAOptions::ModifyCFG));
			fpm.addPass(EarlyCSEPass());
			fpm.run(entry, fam);
		}
		localise_globals(m, entry);
		/* After localise_globals, which expects the -D program's entry
		 * block as it is (folding first left its allocas after their
		 * stores); the O2 pipeline below then drops the dead paths. */
		if (fold_mode) {
			for (auto &v : values)
				fold_values[v.name] = v.bits;
			fold_dynamic(m);
		}
	}

	LLVMContext &ctx = m.getContext();
	/* Function constants: !air.function_constants = { ptr @g, type, name, index, required } */
	std::map<GlobalVariable *, std::string> fc_names;
	if (NamedMDNode *fcs = m.getNamedMetadata("air.function_constants")) {
		for (MDNode *n : fcs->operands()) {
			auto *gv = mdconst::dyn_extract_or_null<GlobalVariable>(n->getOperand(0));
			auto *name = dyn_cast<MDString>(n->getOperand(2));
			if (gv && name)
				fc_names[gv] = name->getString().str();
		}
	}
	if (dynamic)
		fc_names.clear();
	std::map<std::string, uint64_t> given;
	for (auto &v : values)
		given[v.name] = v.bits;

	for (auto &[gv, name] : fc_names) {
		bool defined = given.count(name) != 0;
		std::vector<User *> users(gv->user_begin(), gv->user_end());
		for (User *u : users) {
			if (auto *ci = dyn_cast<CallInst>(u)) {
				Function *callee = ci->getCalledFunction();
				if (callee && callee->getName() == "air.is_function_constant_defined") {
					ci->replaceAllUsesWith(ConstantInt::getBool(ctx, defined));
					ci->eraseFromParent();
				}
			}
		}
		Type *t = gv->getValueType();
		Constant *c;
		if (t->isIntegerTy())
			c = ConstantInt::get(t, defined ? given[name] : 0);
		else if (t->isHalfTy() || t->isFloatTy()) {
			uint64_t bits = defined ? given[name] : 0;
			c = ConstantFP::get(ctx, APFloat(t->isHalfTy() ? APFloat::IEEEhalf() : APFloat::IEEEsingle(),
			    APInt(t->getPrimitiveSizeInBits(), bits)));
		} else if (auto *vt = dyn_cast<FixedVectorType>(t)) {
			/* uchar3 and friends: bits hold the bytes, little endian */
			std::vector<Constant *> elems;
			unsigned eb = vt->getElementType()->getPrimitiveSizeInBits();
			uint64_t bits = defined ? given[name] : 0;
			for (unsigned i = 0; i < vt->getNumElements(); i++)
				elems.push_back(ConstantInt::get(vt->getElementType(), (bits >> (i * eb)) & ((1ull << eb) - 1)));
			c = ConstantVector::get(elems);
		} else {
			fail("function constant " + name + " of unsupported type");
		}
		gv->setInitializer(c);
		gv->setConstant(true);
		gv->setExternallyInitialized(false);
		gv->setSection("");
	}
	/* air.normalize_function_constant_predicate(x) is x != 0 as i8 */
	if (Function *norm = m.getFunction("air.normalize_function_constant_predicate.i8")) {
		std::vector<User *> users(norm->user_begin(), norm->user_end());
		for (User *u : users) {
			auto *ci = cast<CallInst>(u);
			IRBuilder<> b(ci);
			Value *v = b.CreateZExt(b.CreateICmpNE(ci->getArgOperand(0), b.getInt8(0)), b.getInt8Ty());
			ci->replaceAllUsesWith(v);
			ci->eraseFromParent();
		}
	}
	/* Run the static initialiser through GlobalOpt's evaluator. */
	for (Function &f : m)
		if (!dynamic)
		if (f.getSection() == "air.static_init")
			appendToGlobalCtors(m, &f, 0);
	/* Inline everything into the entry point. */
	for (Function &f : m) {
		if (&f == &entry || f.isDeclaration())
			continue;
		f.removeFnAttr(Attribute::NoInline);
		f.addFnAttr(Attribute::AlwaysInline);
	}
	LoopAnalysisManager lam;
	FunctionAnalysisManager fam;
	CGSCCAnalysisManager cam;
	ModuleAnalysisManager mam;
	PassBuilder pb;
	pb.registerModuleAnalyses(mam);
	pb.registerCGSCCAnalyses(cam);
	pb.registerFunctionAnalyses(fam);
	pb.registerLoopAnalyses(lam);
	pb.crossRegisterProxies(lam, fam, cam, mam);
	ModulePassManager mpm;
	mpm.addPass(GlobalOptPass());
	mpm.addPass(AlwaysInlinerPass());
	mpm.addPass(pb.buildPerModuleDefaultPipeline(OptimizationLevel::O2));
	mpm.addPass(GlobalOptPass());
	mpm.run(m, mam);
}

/* ---- translation ---- */

struct ArgInfo {
	std::string kind;               /* air.buffer, air.texture, ... */
	std::string varying;            /* generated(...) name */
	int location = -1;              /* air.location_index */
	int address_space = -1;
	bool perspective = true, flat = false, centroid = false, unused = false;
	int render_target = -1;
};

static std::vector<ArgInfo> parse_args(MDNode *list, bool outputs)
{
	std::vector<ArgInfo> out;
	if (!list)
		return out;
	for (const MDOperand &op : list->operands()) {
		auto *n = dyn_cast<MDNode>(op.get());
		ArgInfo a;
		if (!n) {
			out.push_back(a);
			continue;
		}
		unsigned i = outputs ? 0 : 1;  /* inputs start with the argument index */
		for (; i < n->getNumOperands(); i++) {
			auto *s = dyn_cast_or_null<MDString>(n->getOperand(i).get());
			if (!s)
				continue;
			StringRef k = s->getString();
			/* -1 when the operand is not a constant: binding indices given
			 * by function-constant expressions ([[texture(expr)]], a global
			 * the static initialiser sets; lower_expr_locations turned the
			 * argument's uses into s8.texidx calls), or a global that was
			 * optimised away */
			auto int_at = [&](unsigned j) {
				auto *c = j < n->getNumOperands() ? mdconst::dyn_extract_or_null<ConstantInt>(n->getOperand(j)) : nullptr;
				return c ? (int)c->getZExtValue() : -1;
			};
			if (k == "air.function_constant") {
				i++;    /* predicate node */
			} else if (k == "air.location_index") {
				a.location = int_at(i + 1);
				i += 2;
			} else if (k == "air.address_space") {
				a.address_space = int_at(++i);
			} else if (k == "air.render_target") {
				a.kind = k.str();
				a.render_target = int_at(i + 1);
				i += 2;
			} else if (k == "air.vertex_output" || k == "air.fragment_input") {
				a.kind = k.str();
				a.varying = cast<MDString>(n->getOperand(++i))->getString().str();
			} else if (k == "air.no_perspective") {
				a.perspective = false;
			} else if (k == "air.flat") {
				a.flat = true;
			} else if (k == "air.centroid") {
				a.centroid = true;
			} else if (k == "air.arg_unused") {
				a.unused = true;
			} else if (k == "air.arg_type_name" || k == "air.arg_name" || k == "air.arg_type_size" ||
			    k == "air.arg_type_align_size" || k == "air.buffer_size" || k == "air.struct_type_info" ||
			    k == "air.raster_order_group") {
				i++;
			} else if (k.starts_with("air.") && a.kind.empty() && k != "air.read" && k != "air.write" &&
			    k != "air.read_write" && k != "air.center" && k != "air.perspective" && k != "air.sample") {
				a.kind = k.str();
			}
		}
		out.push_back(a);
	}
	return out;
}

struct InlineSampler {
	uint64_t state;
};

class Translator {
public:
	Translator(Module &m, Function &f, gl_shader_stage stage, const std::vector<std::string> &varyings)
	    : m(m), f(f), dl(m.getDataLayout()), stage(stage), varying_table(varyings)
	{
	}

	nir_shader *run();
	std::vector<InlineSampler> inline_samplers;
	std::vector<std::string> unsupported;
	bool writes_images = false;
	std::vector<uint8_t> constant_data;     /* UBO S8_UBO_DATA */

private:
	Module &m;
	Function &f;
	const DataLayout &dl;
	gl_shader_stage stage;
	std::vector<std::string> varying_table;
	nir_builder b;
	std::map<const Value *, std::vector<nir_def *>> vals;   /* aggregates: one def per member */
	std::map<const BasicBlock *, nir_block *> blocks;
	std::map<const PHINode *, nir_variable *> phis;
	std::map<const AllocaInst *, nir_variable *> allocas;
	std::vector<ArgInfo> args, rets;
	std::map<const GlobalVariable *, int> sampler_index;
	std::map<const GlobalVariable *, uint32_t> data_offset;
	void put_constant(const Constant *c, size_t at);

	nir_def *def(const Value *v);
	std::vector<nir_def *> agg(const Value *v);
	const glsl_type *glsl(Type *t);
	unsigned bits(Type *t);
	void emit(Instruction &i);
	void emit_call(CallInst &ci);
	void emit_air(CallInst &ci, StringRef name);
	void branch_to(const BasicBlock *from, const BasicBlock *to);
	nir_def *gep(const GEPOperator *g);
	int varying_slot(const std::string &name);
	nir_variable *io_var(nir_variable_mode mode, const char *name, const glsl_type *t, int location);
	void store_return(const Value *rv);
};

static nir_alu_type base_type(Type *t)
{
	Type *s = t->getScalarType();
	unsigned bits = s->getPrimitiveSizeInBits();
	if (s->isFloatingPointTy())
		return (nir_alu_type)(nir_type_float | bits);
	if (bits == 1)
		return nir_type_bool1;
	return (nir_alu_type)(nir_type_int | bits);
}

unsigned Translator::bits(Type *t)
{
	Type *s = t->getScalarType();
	if (s->isPointerTy())
		return 64;
	return s->getPrimitiveSizeInBits();
}

const glsl_type *Translator::glsl(Type *t)
{
	unsigned n = 1;
	if (auto *vt = dyn_cast<FixedVectorType>(t)) {
		n = vt->getNumElements();
		t = vt->getElementType();
	}
	enum glsl_base_type bt;
	if (t->isHalfTy())
		bt = GLSL_TYPE_FLOAT16;
	else if (t->isFloatTy())
		bt = GLSL_TYPE_FLOAT;
	else if (t->isIntegerTy(1))
		bt = GLSL_TYPE_BOOL;
	else if (t->isIntegerTy(8))
		bt = GLSL_TYPE_UINT8;
	else if (t->isIntegerTy(16))
		bt = GLSL_TYPE_UINT16;
	else if (t->isIntegerTy(32))
		bt = GLSL_TYPE_UINT;
	else if (t->isIntegerTy(64) || t->isPointerTy())
		bt = GLSL_TYPE_UINT64;
	else
		fail("unsupported type");
	return glsl_vector_type(bt, n);
}

nir_def *Translator::def(const Value *v)
{
	auto it = vals.find(v);
	if (it != vals.end())
		return it->second[0];
	if (isa<UndefValue>(v) || isa<PoisonValue>(v)) {
		Type *t = v->getType();
		unsigned n = isa<FixedVectorType>(t) ? cast<FixedVectorType>(t)->getNumElements() : 1;
		return nir_undef(&b, n, bits(t));
	}
	if (auto *ci = dyn_cast<ConstantInt>(v)) {
		return nir_imm_intN_t(&b, ci->getZExtValue(), ci->getBitWidth());
	}
	if (auto *cf = dyn_cast<ConstantFP>(v)) {
		unsigned w = bits(cf->getType());
		return nir_imm_floatN_t(&b, cf->getValueAPF().convertToDouble(), w);
	}
	if (isa<ConstantAggregateZero>(v) && v->getType()->isVectorTy()) {
		auto *vt = cast<FixedVectorType>(v->getType());
		nir_def *z = vt->getElementType()->isFloatingPointTy() ?
		    nir_imm_floatN_t(&b, 0, bits(vt)) : nir_imm_intN_t(&b, 0, bits(vt));
		return nir_replicate(&b, z, vt->getNumElements());
	}
	if (auto *cv = dyn_cast<ConstantDataVector>(v)) {
		nir_def *c[NIR_MAX_VEC_COMPONENTS];
		for (unsigned i = 0; i < cv->getNumElements(); i++)
			c[i] = def(cv->getElementAsConstant(i));
		return nir_vec(&b, c, cv->getNumElements());
	}
	if (auto *cv = dyn_cast<ConstantVector>(v)) {
		nir_def *c[NIR_MAX_VEC_COMPONENTS];
		for (unsigned i = 0; i < cv->getNumOperands(); i++)
			c[i] = def(cv->getOperand(i));
		return nir_vec(&b, c, cv->getNumOperands());
	}
	if (isa<ConstantPointerNull>(v))
		return nir_imm_int64(&b, 0);
	if (auto *gv = dyn_cast<GlobalVariable>(v)) {
		if (sampler_index.count(gv))
			return nir_imm_int64(&b, sampler_index[gv]);
		if (gv->isConstant() && gv->hasInitializer() && gv->getAddressSpace() == 2) {
			/* constant table: copied into UBO S8_UBO_DATA */
			if (!data_offset.count(gv)) {
				size_t at = (constant_data.size() + 15) & ~(size_t)15;
				constant_data.resize(at + dl.getTypeAllocSize(gv->getValueType()));
				put_constant(gv->getInitializer(), at);
				data_offset[gv] = (uint32_t)at;
			}
			return nir_imm_int64(&b, ((uint64_t)S8_UBO_DATA << 32) | data_offset[gv]);
		}
	}
	if (auto *g = dyn_cast<GEPOperator>(v))
		return gep(g);
	if (auto *ce = dyn_cast<ConstantExpr>(v))
		if (ce->isCast())
			return def(ce->getOperand(0));
	std::string s;
	raw_string_ostream os(s);
	v->print(os);
	fail("no value for " + s);
	return nullptr;
}

std::vector<nir_def *> Translator::agg(const Value *v)
{
	auto it = vals.find(v);
	if (it != vals.end())
		return it->second;
	if (auto *st = dyn_cast<StructType>(v->getType())) {
		std::vector<nir_def *> out;
		for (unsigned i = 0; i < st->getNumElements(); i++) {
			Type *e = st->getElementType(i);
			unsigned n = isa<FixedVectorType>(e) ? cast<FixedVectorType>(e)->getNumElements() : 1;
			if (auto *c = dyn_cast<Constant>(v))
				out.push_back(def(c->getAggregateElement(i)));
			else
				out.push_back(nir_undef(&b, n, bits(e)));
		}
		return out;
	}
	return { def(v) };
}

int Translator::varying_slot(const std::string &name)
{
	/* entries are "name" (slot = position) or "name=slot" */
	for (size_t i = 0; i < varying_table.size(); i++) {
		const std::string &e = varying_table[i];
		size_t eq = e.rfind('=');
		if (eq != std::string::npos && e.compare(0, eq, name) == 0 && eq == name.size())
			return VARYING_SLOT_VAR0 + atoi(e.c_str() + eq + 1);
		if (e == name)
			return VARYING_SLOT_VAR0 + (int)i;
	}
	varying_table.push_back(name);
	fprintf(stderr, "air2nir: note: varying %s not in the table, slot VAR%zu\n", name.c_str(),
	    varying_table.size() - 1);
	return VARYING_SLOT_VAR0 + (int)varying_table.size() - 1;
}

nir_variable *Translator::io_var(nir_variable_mode mode, const char *name, const glsl_type *t, int location)
{
	nir_variable *v = nir_variable_create(b.shader, mode, t, name);
	v->data.location = location;
	v->data.driver_location = location;
	return v;
}

/* Pointers into Metal buffers are 64-bit values: device buffers (address
 * space 1) hold the GPU address; constant buffers (address space 2) hold
 * the UBO index in the high word and the byte offset in the low word, so
 * pointer arithmetic, selects and phis work unchanged. */
nir_def *Translator::gep(const GEPOperator *g)
{
	nir_def *base = def(g->getPointerOperand());
	Type *ty = g->getSourceElementType();
	nir_def *off = nir_imm_int64(&b, 0);
	auto idx = g->idx_begin();
	off = nir_iadd(&b, off, nir_imul_imm(&b, nir_i2i64(&b, def(*idx)), dl.getTypeAllocSize(ty)));
	Type *cur = ty;
	for (++idx; idx != g->idx_end(); ++idx) {
		if (auto *st = dyn_cast<StructType>(cur)) {
			unsigned fi = cast<ConstantInt>(*idx)->getZExtValue();
			off = nir_iadd_imm(&b, off, dl.getStructLayout(st)->getElementOffset(fi));
			cur = st->getElementType(fi);
		} else {
			Type *elem = cur->isArrayTy() ? cur->getArrayElementType() :
			    cast<VectorType>(cur)->getElementType();
			off = nir_iadd(&b, off, nir_imul_imm(&b, nir_i2i64(&b, def(*idx)), dl.getTypeAllocSize(elem)));
			cur = elem;
		}
	}
	return nir_iadd(&b, base, off);
}

void Translator::branch_to(const BasicBlock *from, const BasicBlock *to)
{
	for (const PHINode &phi : to->phis()) {
		const Value *in = phi.getIncomingValueForBlock(from);
		nir_store_var(&b, phis.at(&phi), def(in), nir_component_mask(phi.getType()->isVectorTy() ?
		    cast<FixedVectorType>(phi.getType())->getNumElements() : 1));
	}
}

static nir_def *nir_fsat_any(nir_builder *b, nir_def *x) { return nir_fsat(b, x); }

void Translator::put_constant(const Constant *c, size_t at)
{
	if (isa<ConstantAggregateZero>(c) || isa<UndefValue>(c))
		return;
	if (auto *s = dyn_cast<ConstantDataSequential>(c)) {
		StringRef raw = s->getRawDataValues();
		memcpy(&constant_data[at], raw.data(), raw.size());
		return;
	}
	if (auto *ci = dyn_cast<ConstantInt>(c)) {
		uint64_t v = ci->getZExtValue();
		memcpy(&constant_data[at], &v, dl.getTypeStoreSize(ci->getType()));
		return;
	}
	if (auto *cf = dyn_cast<ConstantFP>(c)) {
		uint64_t v = cf->getValueAPF().bitcastToAPInt().getZExtValue();
		memcpy(&constant_data[at], &v, dl.getTypeStoreSize(cf->getType()));
		return;
	}
	if (auto *cs = dyn_cast<ConstantStruct>(c)) {
		const StructLayout *sl = dl.getStructLayout(cs->getType());
		for (unsigned i = 0; i < cs->getNumOperands(); i++)
			put_constant(cs->getOperand(i), at + sl->getElementOffset(i));
		return;
	}
	if (isa<ConstantArray>(c) || isa<ConstantVector>(c)) {
		Type *et = isa<ConstantArray>(c) ? cast<ArrayType>(c->getType())->getElementType()
		    : cast<FixedVectorType>(c->getType())->getElementType();
		uint64_t stride = dl.getTypeAllocSize(et);
		for (unsigned i = 0; i < c->getNumOperands(); i++)
			put_constant(cast<Constant>(c->getOperand(i)), at + i * stride);
		return;
	}
	std::string s;
	raw_string_ostream os(s);
	c->print(os);
	fail("unsupported constant initialiser " + s);
}

nir_shader *Translator::run()
{
	b = nir_builder_init_simple_shader(stage, pan_shader_get_compiler_options(6), "%s",
	    f.getName().str().c_str());
	b.shader->info.internal = false;

	/* argument and return metadata */
	const char *md = stage == MESA_SHADER_VERTEX ? "air.vertex" : "air.fragment";
	MDNode *entry = nullptr;
	if (NamedMDNode *nm = m.getNamedMetadata(md))
		for (MDNode *n : nm->operands())
			if (mdconst::dyn_extract_or_null<Function>(n->getOperand(0)) == &f)
				entry = n;
	if (!entry)
		fail("no " + std::string(md) + " metadata");
	rets = parse_args(dyn_cast<MDNode>(entry->getOperand(1)), true);
	args = parse_args(dyn_cast<MDNode>(entry->getOperand(2)), false);

	/* inline constexpr samplers */
	if (NamedMDNode *ss = m.getNamedMetadata("air.sampler_states")) {
		for (MDNode *n : ss->operands()) {
			auto *gv = mdconst::dyn_extract_or_null<GlobalVariable>(n->getOperand(1));
			if (gv && gv->hasInitializer()) {
				uint64_t st = 0;
				if (auto *ci = dyn_cast<ConstantInt>(gv->getInitializer()))
					st = ci->getZExtValue();
				sampler_index[gv] = S8_INLINE_SAMPLER_BASE + (int)inline_samplers.size();
				inline_samplers.push_back({ st });
			}
		}
	}

	nir_function_impl *impl = b.impl;
	impl->structured = false;

	/* entry: load the inputs */
	for (Argument &arg : f.args()) {
		const ArgInfo &a = args[arg.getArgNo()];
		Type *t = arg.getType();
		if (arg.use_empty())
			continue;
		if (a.kind == "air.vertex_input") {
			nir_variable *v = io_var(nir_var_shader_in, "attr", glsl(t), VERT_ATTRIB_GENERIC0 + a.location);
			/* The Bifrost compiler indexes the attribute table with
			 * driver_location: Metal attribute L is entry L of the
			 * table s8mtl_draw.c builds (as PanVK, panvk_vX_shader.c).
			 * With VERT_ATTRIB_GENERIC0 + L (15 + L) every vertex job
			 * read descriptors past the table and hung the G71. */
			v->data.driver_location = a.location;
			vals[&arg] = { nir_load_var(&b, v) };
		} else if (a.kind == "air.fragment_input") {
			nir_variable *v = io_var(nir_var_shader_in, a.varying.c_str(), glsl(t), varying_slot(a.varying));
			v->data.interpolation = a.flat ? INTERP_MODE_FLAT :
			    (a.perspective ? INTERP_MODE_SMOOTH : INTERP_MODE_NOPERSPECTIVE);
			v->data.centroid = a.centroid;
			vals[&arg] = { nir_load_var(&b, v) };
		} else if (a.kind == "air.position" && stage == MESA_SHADER_FRAGMENT) {
			nir_variable *v = io_var(nir_var_system_value, "frag_coord", glsl_vec4_type(), SYSTEM_VALUE_FRAG_COORD);
			(void)v;
			vals[&arg] = { nir_load_frag_coord(&b) };
		} else if (a.kind == "air.render_target" && a.render_target > 0) {
			/* Passes have one colour attachment (s8mtl_draw.c), and a
			 * tile read of a render target the framebuffer descriptor
			 * lacks faulted INSTR_INVALID_ENC on the G71 (LD_TILE rt 1 in
			 * fixed_frag_lph_cpf, found by the loader bisection): colour
			 * attachments past 0 read as zero. */
			const glsl_type *gt = glsl(t);
			vals[&arg] = { nir_imm_zero(&b, glsl_get_vector_elements(gt), glsl_get_bit_size(gt)) };
		} else if (a.kind == "air.render_target") {
			/* framebuffer fetch */
			nir_variable *v = io_var(nir_var_shader_out, "fb", glsl(t), FRAG_RESULT_DATA0 + a.render_target);
			v->data.fb_fetch_output = true;
			b.shader->info.outputs_read |= BITFIELD64_BIT(FRAG_RESULT_DATA0 + a.render_target);
			b.shader->info.fs.uses_fbfetch_output = true;
			vals[&arg] = { nir_load_var(&b, v) };
		} else if (a.kind == "air.vertex_id") {
			vals[&arg] = { nir_load_vertex_id(&b) };
		} else if (a.kind == "air.instance_id") {
			vals[&arg] = { nir_load_instance_id(&b) };
		} else if (a.kind == "air.front_facing") {
			vals[&arg] = { nir_load_front_face(&b, 1) };
		} else if (a.kind == "air.point_coord") {
			vals[&arg] = { nir_load_point_coord(&b) };
		} else if (a.kind == "air.texture" || a.kind == "air.sampler") {
			/* texture and sampler "pointers" are their binding indices, so
			 * a shader may select one at run time (texture_offset) */
			vals[&arg] = { nir_imm_int64(&b, a.location) };
		} else if (a.kind == "air.buffer") {
			if (a.location >= S8_UBO_DATA)
				unsupported.push_back("buffer index " + std::to_string(a.location) + " (reserved UBO)");
			if (a.address_space == 2)
				vals[&arg] = { nir_imm_int64(&b, (uint64_t)a.location << 32) };
			else
				vals[&arg] = { s8_load_ubo(&b, 1, 64, nir_imm_int(&b, S8_UBO_ADDRS),
				    nir_imm_int(&b, a.location * 8), 8) };
		} else {
			unsupported.push_back("argument kind " + a.kind);
		}
	}

	/* phis become local variables */
	for (BasicBlock &bb : f)
		for (PHINode &phi : bb.phis())
			phis[&phi] = nir_local_variable_create(impl, glsl(phi.getType()), "phi");
	for (BasicBlock &bb : f)
		for (Instruction &i : bb)
			if (auto *al = dyn_cast<AllocaInst>(&i))
				allocas[al] = nir_local_variable_create(impl, glsl_array_type(glsl_uint8_t_type(),
				    dl.getTypeAllocSize(al->getAllocatedType()), 1), "alloca");

	/* blocks in LLVM order; the entry is NIR's start block */
	bool first = true;
	for (BasicBlock &bb : f) {
		if (first) {
			blocks[&bb] = nir_start_block(impl);
			first = false;
			continue;
		}
		nir_block *n = nir_block_create(b.shader);
		exec_list_push_tail(&impl->body, &n->cf_node.node);
		n->cf_node.parent = &impl->cf_node;
		blocks[&bb] = n;
	}
	/* Instructions are translated in reverse post-order, so a value is
	 * translated before its uses also when a loop's latch comes before its
	 * header in the function's block list (VideoToolbox's shaders: "no
	 * value for"); blocks keep their NIR positions. Unreachable blocks
	 * last. */
	std::vector<BasicBlock *> order;
	std::set<BasicBlock *> seen;
	for (BasicBlock *bb : ReversePostOrderTraversal<Function *>(&f)) {
		order.push_back(bb);
		seen.insert(bb);
	}
	for (BasicBlock &bb : f)
		if (!seen.count(&bb))
			order.push_back(&bb);
	for (BasicBlock *bbp : order) {
		BasicBlock &bb = *bbp;
		b.cursor = nir_after_block(blocks[&bb]);
		for (PHINode &phi : bb.phis())
			vals[&phi] = { nir_load_var(&b, phis[&phi]) };
		for (Instruction &i : bb)
			if (!isa<PHINode>(&i))
				emit(i);
	}
	nir_lower_goto_ifs(b.shader);
	nir_validate_shader(b.shader, "air2nir");
	return b.shader;
}

void Translator::store_return(const Value *rv)
{
	std::vector<nir_def *> parts;
	if (rv->getType()->isStructTy())
		parts = agg(rv);
	else
		parts = { def(rv) };
	unsigned j = 0;
	for (size_t i = 0; i < rets.size() && j < parts.size(); i++, j++) {
		const ArgInfo &r = rets[i];
		nir_def *v = parts[j];
		const glsl_type *t = glsl_vector_type(v->bit_size == 16 ? GLSL_TYPE_FLOAT16 :
		    (v->bit_size == 1 ? GLSL_TYPE_BOOL : GLSL_TYPE_FLOAT), v->num_components);
		nir_variable *var = nullptr;
		if (r.kind == "air.render_target" && r.render_target > 0)
			continue;       /* one colour attachment: no writes past 0 (no BLEND for RT > 0) */
		if (r.kind == "air.position")
			var = io_var(nir_var_shader_out, "position", glsl_vec4_type(), VARYING_SLOT_POS);
		else if (r.kind == "air.vertex_output")
			var = io_var(nir_var_shader_out, r.varying.c_str(), t, varying_slot(r.varying));
		else if (r.kind == "air.render_target")
			var = io_var(nir_var_shader_out, "color", t, FRAG_RESULT_DATA0 + r.render_target);
		else if (r.kind == "air.point_size")
			var = io_var(nir_var_shader_out, "psiz", glsl_float_type(), VARYING_SLOT_PSIZ);
		else if (r.kind == "air.depth")
			var = io_var(nir_var_shader_out, "depth", glsl_float_type(), FRAG_RESULT_DEPTH);
		else {
			unsupported.push_back("return kind " + r.kind);
			continue;
		}
		if (r.kind == "air.position" && v->bit_size != 32)
			v = nir_f2f32(&b, v);
		nir_store_var(&b, var, v, nir_component_mask(v->num_components));
	}
}

/* Elementwise helpers for the AIR math intrinsics. */
typedef nir_def *(*unop)(nir_builder *, nir_def *);
typedef nir_def *(*binop)(nir_builder *, nir_def *, nir_def *);
typedef nir_def *(*triop)(nir_builder *, nir_def *, nir_def *, nir_def *);

static nir_def *s8_fmix(nir_builder *b, nir_def *x, nir_def *y, nir_def *a) { return nir_flrp(b, x, y, a); }
/* nir_image_store's macro mixes positional and designated initialisers,
 * which C++ rejects: build the indices by hand. */
static nir_def *s8_image_store(nir_builder *b, nir_def *idx, nir_def *coord, nir_def *v, nir_alu_type t)
{
	struct _nir_image_store_indices ix = {};
	ix.image_dim = GLSL_SAMPLER_DIM_2D;
	ix.image_array = false;
	ix.format = PIPE_FORMAT_NONE;
	ix.access = ACCESS_NON_READABLE;
	ix.src_type = t;
	_nir_build_image_store(b, idx, coord, nir_imm_int(b, 0), v, nir_imm_int(b, 0), ix);
	return NULL;
}

static nir_def *s8_fclamp(nir_builder *b, nir_def *x, nir_def *lo, nir_def *hi) { return nir_fmin(b, nir_fmax(b, x, lo), hi); }
static nir_def *s8_powr(nir_builder *b, nir_def *x, nir_def *y) { return nir_fpow(b, x, y); }
static nir_def *s8_log_e(nir_builder *b, nir_def *x) { return nir_fmul_imm(b, nir_flog2(b, x), 0.69314718055994530942); }
static nir_def *s8_exp_e(nir_builder *b, nir_def *x) { return nir_fexp2(b, nir_fmul_imm(b, x, 1.44269504088896340736)); }
static nir_def *s8_sign(nir_builder *b, nir_def *x) { return nir_fsign(b, x); }
static nir_def *s8_rsqrt(nir_builder *b, nir_def *x) { return nir_frsq(b, x); }
static nir_def *s8_step(nir_builder *b, nir_def *edge, nir_def *x) { return nir_b2fN(b, nir_fge(b, x, edge), x->bit_size); }
static nir_def *s8_smoothstep(nir_builder *b, nir_def *e0, nir_def *e1, nir_def *x)
{
	nir_def *t = nir_fsat(b, nir_fdiv(b, nir_fsub(b, x, e0), nir_fsub(b, e1, e0)));
	return nir_fmul(b, nir_fmul(b, t, t), nir_fsub(b, nir_imm_floatN_t(b, 3.0, x->bit_size),
	    nir_fmul_imm(b, t, 2.0)));
}
static nir_def *s8_fma(nir_builder *b, nir_def *x, nir_def *y, nir_def *z) { return nir_ffma(b, x, y, z); }
static nir_def *s8_frac(nir_builder *b, nir_def *x) { return nir_ffract(b, x); }
static nir_def *s8_fwidth(nir_builder *b, nir_def *x) { return nir_fadd(b, nir_fabs(b, nir_ddx(b, x)), nir_fabs(b, nir_ddy(b, x))); }
static nir_def *s8_ddx(nir_builder *b, nir_def *x) { return nir_ddx(b, x); }
static nir_def *s8_ddy(nir_builder *b, nir_def *x) { return nir_ddy(b, x); }
static nir_def *s8_normalize(nir_builder *b, nir_def *x) { return nir_fmul(b, x, nir_frsq(b, nir_fdot(b, x, x))); }
static nir_def *s8_length(nir_builder *b, nir_def *x) { return nir_fsqrt(b, nir_fdot(b, x, x)); }
static nir_def *s8_dot(nir_builder *b, nir_def *x, nir_def *y) { return nir_fdot(b, x, y); }
static nir_def *s8_any(nir_builder *b, nir_def *x) { return nir_bany(b, x); }
static nir_def *s8_all(nir_builder *b, nir_def *x) { return nir_ball(b, x); }
static nir_def *s8_tan(nir_builder *b, nir_def *x) { return nir_fdiv(b, nir_fsin(b, x), nir_fcos(b, x)); }
static nir_def *s8_fmax3(nir_builder *b, nir_def *x, nir_def *y, nir_def *z) { return nir_fmax(b, nir_fmax(b, x, y), z); }
static nir_def *s8_fmin3(nir_builder *b, nir_def *x, nir_def *y, nir_def *z) { return nir_fmin(b, nir_fmin(b, x, y), z); }
static nir_def *s8_fmedian3(nir_builder *b, nir_def *x, nir_def *y, nir_def *z)
{
	return nir_fmax(b, nir_fmin(b, x, y), nir_fmin(b, nir_fmax(b, x, y), z));
}

void Translator::emit_air(CallInst &ci, StringRef name)
{
	StringRef base = name.drop_front(4);     /* after "air." */
	base = base.take_until([](char c) { return c == '.'; });
	bool fast = base.consume_front("fast_");
	(void)fast;
	auto arg = [&](unsigned i) { return def(ci.getArgOperand(i)); };
	nir_def *r = nullptr;

	static const std::map<std::string, unop> unops = {
		{ "fabs", nir_fabs }, { "sqrt", nir_fsqrt }, { "rsqrt", s8_rsqrt }, { "floor", nir_ffloor },
		{ "ceil", nir_fceil }, { "trunc", nir_ftrunc }, { "rint", nir_fround_even }, { "round", nir_fround_even },
		{ "fract", s8_frac }, { "saturate", nir_fsat_any }, { "exp2", nir_fexp2 }, { "log2", nir_flog2 },
		{ "exp", s8_exp_e }, { "log", s8_log_e }, { "sin", nir_fsin }, { "cos", nir_fcos }, { "sign", s8_sign },
		{ "fwidth", s8_fwidth }, { "dfdx", s8_ddx }, { "dfdy", s8_ddy }, { "normalize", s8_normalize },
		{ "length", s8_length }, { "any", s8_any }, { "all", s8_all }, { "atan", nir_atan },
		{ "tan", s8_tan },
	};
	static const std::map<std::string, binop> binops = {
		{ "fmin", nir_fmin }, { "fmax", nir_fmax }, { "pow", nir_fpow }, { "powr", s8_powr },
		{ "dot", s8_dot }, { "step", s8_step }, { "min", nir_fmin }, { "max", nir_fmax },
		{ "atan2", nir_atan2 },
	};
	static const std::map<std::string, triop> triops = {
		{ "mix", s8_fmix }, { "clamp", s8_fclamp }, { "fma", s8_fma }, { "smoothstep", s8_smoothstep },
		{ "fmax3", s8_fmax3 }, { "fmin3", s8_fmin3 }, { "fmedian3", s8_fmedian3 },
		{ "max3", s8_fmax3 }, { "min3", s8_fmin3 }, { "median3", s8_fmedian3 },
	};

	std::string bs = base.str();
	if (unops.count(bs)) {
		r = unops.at(bs)(&b, arg(0));
	} else if (binops.count(bs)) {
		r = binops.at(bs)(&b, arg(0), arg(1));
	} else if (triops.count(bs)) {
		r = triops.at(bs)(&b, arg(0), arg(1), arg(2));
	} else if (bs == "unpack" || bs == "pack") {
		/* air.unpack.unorm4x8.v4f16 and friends */
		StringRef kind = name.drop_front(4 + bs.size() + 1).take_until([](char c) { return c == '.'; });
		nir_def *x = arg(0);
		if (bs == "unpack" && kind == "unorm4x8") r = nir_unpack_unorm_4x8(&b, x);
		else if (bs == "unpack" && kind == "snorm4x8") r = nir_unpack_snorm_4x8(&b, x);
		else if (bs == "unpack" && kind == "unorm2x16") r = nir_unpack_unorm_2x16(&b, x);
		else if (bs == "unpack" && kind == "snorm2x16") r = nir_unpack_snorm_2x16(&b, x);
		else if (bs == "pack" && kind == "unorm4x8") r = nir_pack_unorm_4x8(&b, nir_f2f32(&b, x));
		else if (bs == "pack" && kind == "snorm4x8") r = nir_pack_snorm_4x8(&b, nir_f2f32(&b, x));
		else if (bs == "pack" && kind == "unorm2x16") r = nir_pack_unorm_2x16(&b, nir_f2f32(&b, x));
		else if (bs == "pack" && kind == "snorm2x16") r = nir_pack_snorm_2x16(&b, nir_f2f32(&b, x));
		else if (bs == "pack" && kind == "unorm" && name.contains(".rgb565.")) {
			/* air.pack.unorm.rgb565: red in the top 5 bits (B5G6R5 memory order) */
			nir_def *f = nir_fsat(&b, nir_f2f32(&b, x));
			nir_def *q = nir_f2u32(&b, nir_fround_even(&b, nir_fmul(&b, f, nir_imm_vec3(&b, 31.0, 63.0, 31.0))));
			nir_def *v = nir_ior(&b, nir_ishl_imm(&b, nir_channel(&b, q, 0), 11),
			    nir_ior(&b, nir_ishl_imm(&b, nir_channel(&b, q, 1), 5), nir_channel(&b, q, 2)));
			r = nir_u2uN(&b, v, bits(ci.getType()));
		} else if (bs == "unpack" && kind == "unorm" && name.contains(".rgb565.")) {
			nir_def *v = nir_u2u32(&b, x);
			nir_def *q = nir_vec3(&b, nir_ubitfield_extract_imm(&b, v, 11, 5),
			    nir_ubitfield_extract_imm(&b, v, 5, 6), nir_ubitfield_extract_imm(&b, v, 0, 5));
			r = nir_fmul(&b, nir_u2f32(&b, q), nir_imm_vec3(&b, 1.0 / 31, 1.0 / 63, 1.0 / 31));
		} else if (bs == "pack" && kind == "unorm" && name.contains(".rgb10a2.")) {
			/* air.pack.unorm.rgb10a2 (pack_float_to_unorm1010102): red in
			 * bits 0-9, green 10-19, blue 20-29, alpha 30-31 */
			nir_def *f = nir_fsat(&b, nir_f2f32(&b, x));
			nir_def *q = nir_f2u32(&b, nir_fround_even(&b, nir_fmul(&b, f, nir_imm_vec4(&b, 1023.0, 1023.0, 1023.0, 3.0))));
			nir_def *v = nir_ior(&b, nir_ior(&b, nir_channel(&b, q, 0), nir_ishl_imm(&b, nir_channel(&b, q, 1), 10)),
			    nir_ior(&b, nir_ishl_imm(&b, nir_channel(&b, q, 2), 20), nir_ishl_imm(&b, nir_channel(&b, q, 3), 30)));
			r = nir_u2uN(&b, v, bits(ci.getType()));
		} else if (bs == "unpack" && kind == "unorm" && name.contains(".rgb10a2.")) {
			nir_def *v = nir_u2u32(&b, x);
			nir_def *q = nir_vec4(&b, nir_ubitfield_extract_imm(&b, v, 0, 10), nir_ubitfield_extract_imm(&b, v, 10, 10),
			    nir_ubitfield_extract_imm(&b, v, 20, 10), nir_ubitfield_extract_imm(&b, v, 30, 2));
			r = nir_fmul(&b, nir_u2f32(&b, q), nir_imm_vec4(&b, 1.0 / 1023, 1.0 / 1023, 1.0 / 1023, 1.0 / 3));
		}
		else
			unsupported.push_back(name.str());
	} else if (bs == "convert") {
		/* air.convert.<dst kind>.<dst type>.<src kind>.<src type> */
		SmallVector<StringRef, 6> p;
		name.split(p, '.');
		Type *dt = ci.getType();
		nir_def *x = arg(0);
		char dk = p[2][0], sk = p[4][0];
		nir_alu_type src_t = sk == 'f' ? nir_type_float : (sk == 's' ? nir_type_int : nir_type_uint);
		nir_alu_type dst_t = dk == 'f' ? nir_type_float : (dk == 's' ? nir_type_int : nir_type_uint);
		if (x->bit_size == 1)
			x = nir_b2iN(&b, x, 32), src_t = nir_type_uint;
		unsigned db = bits(dt);
		if (db == 1)
			r = nir_ine_imm(&b, x, 0);
		else
			r = nir_type_convert(&b, x, (nir_alu_type)(src_t | x->bit_size), (nir_alu_type)(dst_t | db),
			    nir_rounding_mode_undef);
	} else if (bs == "sample_texture_2d" || bs == "sample_texture_3d" || bs == "sample_texture_cube" ||
	    bs == "read_texture_2d" || bs == "sample_texture_2d_array" || bs == "sample_texture_1d") {
		/* 1D textures are Wx1 2D textures for the driver: sampled at y = 0.5 */
		bool one_d = bs == "sample_texture_1d";
		bool read = bs.starts_with("read");
		nir_def *tex_idx = nir_u2u32(&b, def(ci.getArgOperand(0)));
		nir_def *samp_idx = read ? nir_imm_int(&b, 0) : nir_u2u32(&b, def(ci.getArgOperand(1)));
		unsigned dim_n = bs == "sample_texture_3d" || bs == "sample_texture_cube" ? 3 : 2;
		nir_def *coord = arg(read ? 1 : 2);
		if (one_d)
			coord = nir_vec2(&b, nir_f2f32(&b, nir_channel(&b, coord, 0)), nir_imm_float(&b, 0.5f));
		/* sample_texture_*(tex, sampler, coord, i1, offset, i1 explicit_lod,
		 * float bias_or_lod, float min_lod, i32 access) */
		nir_texop op = nir_texop_tex;
		nir_def *lod = nullptr, *offset = nullptr;
		if (read) {
			op = nir_texop_txf;
			lod = nir_i2i32(&b, arg(2));
		} else {
			auto *explicit_lod = dyn_cast<ConstantInt>(ci.getArgOperand(5));
			const Value *bv = ci.getArgOperand(6);
			bool zero_bias = isa<ConstantFP>(bv) && cast<ConstantFP>(bv)->isZero();
			if (explicit_lod && explicit_lod->isOne()) {
				op = nir_texop_txl;
				lod = nir_f2f32(&b, def(bv));
			} else if (!zero_bias) {
				op = nir_texop_txb;
				lod = nir_f2f32(&b, def(bv));
			}
			if (one_d) {
				if (!isa<ConstantInt>(ci.getArgOperand(4)) || !cast<ConstantInt>(ci.getArgOperand(4))->isZero())
					offset = nir_vec2(&b, nir_i2i32(&b, nir_channel(&b, arg(4), 0)), nir_imm_int(&b, 0));
			} else if (!isa<ConstantAggregateZero>(ci.getArgOperand(4)))
				offset = nir_i2i32(&b, arg(4));
			if (stage != MESA_SHADER_FRAGMENT && op == nir_texop_tex) {
				op = nir_texop_txl;
				lod = nir_imm_float(&b, 0);
			}
		}
		bool tex_const = nir_src_is_const(nir_src_for_ssa(tex_idx));
		bool samp_const = nir_src_is_const(nir_src_for_ssa(samp_idx));
		unsigned nsrc = 1 + (lod ? 1 : 0) + (offset ? 1 : 0) + (tex_const ? 0 : 1) + (samp_const ? 0 : 1);
		nir_tex_instr *tex = nir_tex_instr_create(b.shader, nsrc);
		tex->op = op;
		tex->sampler_dim = bs == "sample_texture_3d" ? GLSL_SAMPLER_DIM_3D :
		    (bs == "sample_texture_cube" ? GLSL_SAMPLER_DIM_CUBE : GLSL_SAMPLER_DIM_2D);
		tex->is_array = bs == "sample_texture_2d_array";
		tex->texture_index = tex_const ? nir_src_as_uint(nir_src_for_ssa(tex_idx)) : 0;
		tex->sampler_index = samp_const ? nir_src_as_uint(nir_src_for_ssa(samp_idx)) : 0;
		auto *st = cast<StructType>(ci.getType());
		Type *rt = st->getElementType(0);
		unsigned rb = bits(rt);
		tex->dest_type = (nir_alu_type)(rt->getScalarType()->isFloatingPointTy() ? (nir_type_float | rb) : (nir_type_uint | rb));
		unsigned k = 0;
		tex->src[k++] = nir_tex_src_for_ssa(nir_tex_src_coord, read ? nir_i2i32(&b, coord) : nir_f2f32(&b, coord));
		if (lod)
			tex->src[k++] = nir_tex_src_for_ssa(op == nir_texop_txb ? nir_tex_src_bias : nir_tex_src_lod, lod);
		if (offset)
			tex->src[k++] = nir_tex_src_for_ssa(nir_tex_src_offset, offset);
		if (!tex_const)
			tex->src[k++] = nir_tex_src_for_ssa(nir_tex_src_texture_offset, tex_idx);
		if (!samp_const)
			tex->src[k++] = nir_tex_src_for_ssa(nir_tex_src_sampler_offset, samp_idx);
		tex->coord_components = dim_n + (tex->is_array ? 1 : 0);
		nir_def_init(&tex->instr, &tex->def, 4, rb);
		nir_builder_instr_insert(&b, &tex->instr);
		vals[&ci] = { &tex->def, nir_imm_intN_t(&b, 0, 8) };
		return;
	} else if (bs == "write_texture_2d") {
		/* air.write_texture_2d[.<coord type>][.u|.s].<value type>(texture,
		 * coord, value, lod, access): an image store. On Bifrost v6 images
		 * are attribute records (LEA_ATTR_TEX in the primary table, index =
		 * the Metal texture index; the driver fills the fragment stage's
		 * attribute table from the bound textures) and ST_CVT takes 32-bit
		 * values only. */
		nir_def *idx = nir_u2u32(&b, def(ci.getArgOperand(0)));
		nir_def *c = nir_u2u32(&b, def(ci.getArgOperand(1)));
		nir_def *coord = nir_vec4(&b, nir_channel(&b, c, 0), nir_channel(&b, c, 1), nir_imm_int(&b, 0),
		    nir_imm_int(&b, 0));
		nir_def *v = def(ci.getArgOperand(2));
		Type *vt = ci.getArgOperand(2)->getType()->getScalarType();
		bool is_float = vt->isFloatingPointTy();
		bool is_signed = name.contains(".s.v");
		if (is_float)
			v = nir_f2f32(&b, v);
		else
			v = is_signed ? nir_i2i32(&b, v) : nir_u2u32(&b, v);
		if (v->num_components < 4) {
			nir_def *ch[4];
			for (unsigned k = 0; k < 4; k++)
				ch[k] = k < v->num_components ? nir_channel(&b, v, k) :
				    (is_float ? nir_imm_float(&b, k == 3 ? 1.0f : 0.0f) : nir_imm_int(&b, k == 3 ? 1 : 0));
			v = nir_vec(&b, ch, 4);
		}
		s8_image_store(&b, idx, coord, v, is_float ? nir_type_float32 : (is_signed ? nir_type_int32 : nir_type_uint32));
		writes_images = true;
		return;
	} else if (bs.starts_with("get_null_texture")) {
		vals[&ci] = { nir_imm_int64(&b, S8_NULL_TEXTURE) };
		return;
	} else if (bs == "get_null_sampler") {
		vals[&ci] = { nir_imm_int64(&b, 0) };
		return;
	} else if (bs == "discard_fragment") {
		nir_terminate(&b);
		return;
	} else if (bs == "get_width_texture_2d" || bs == "get_height_texture_2d") {
		nir_def *tidx = nir_u2u32(&b, arg(0));
		bool tc = nir_src_is_const(nir_src_for_ssa(tidx));
		nir_tex_instr *tex = nir_tex_instr_create(b.shader, tc ? 1 : 2);
		tex->op = nir_texop_txs;
		tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
		tex->texture_index = tc ? nir_src_as_uint(nir_src_for_ssa(tidx)) : 0;
		if (!tc)
			tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_texture_offset, tidx);
		tex->dest_type = nir_type_int32;
		tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_i2i32(&b, arg(1)));
		nir_def_init(&tex->instr, &tex->def, 2, 32);
		nir_builder_instr_insert(&b, &tex->instr);
		r = nir_channel(&b, &tex->def, bs == "get_width_texture_2d" ? 0 : 1);
	} else {
		unsupported.push_back(name.str());
		Type *t = ci.getType();
		if (!t->isVoidTy()) {
			if (t->isStructTy())
				vals[&ci] = agg(UndefValue::get(t));
			else
				vals[&ci] = { nir_undef(&b, isa<FixedVectorType>(t) ? cast<FixedVectorType>(t)->getNumElements() : 1, bits(t)) };
		}
		return;
	}
	if (r) {
		Type *t = ci.getType();
		if (!t->isVoidTy() && r->bit_size != bits(t) && bits(t) != 1)
			r = nir_type_convert(&b, r, (nir_alu_type)(nir_type_float | r->bit_size),
			    (nir_alu_type)(nir_type_float | bits(t)), nir_rounding_mode_undef);
		vals[&ci] = { r };
	}
}

void Translator::emit_call(CallInst &ci)
{
	Function *callee = ci.getCalledFunction();
	if (!callee)
		fail("indirect call");
	StringRef name = callee->getName();
	if (name.starts_with("air.")) {
		emit_air(ci, name);
		return;
	}
	if (name == "s8.texidx") {
		/* a texture whose binding index a function-constant expression
		 * gives (lower_expr_locations): the index, as texture "pointers" */
		vals[&ci] = { nir_u2u64(&b, def(ci.getArgOperand(0))) };
		return;
	}
	if (name == "s8.fcdef" || name.starts_with("s8.fc.")) {
		unsigned index = (unsigned)cast<ConstantInt>(ci.getArgOperand(0))->getZExtValue();
		nir_def *ubo = nir_imm_int(&b, S8_UBO_FC);
		if (name == "s8.fcdef") {
			nir_def *d = s8_load_ubo(&b, 1, 32, ubo, nir_imm_int(&b, index * 16 + 8), 4);
			vals[&ci] = { nir_ine_imm(&b, d, 0) };
			return;
		}
		Type *t = ci.getType();
		auto *vt = dyn_cast<FixedVectorType>(t);
		unsigned n = vt ? vt->getNumElements() : 1, eb = bits(vt ? vt->getElementType() : t);
		nir_def *r;
		if (eb == 64) {
			r = s8_load_ubo(&b, n, 64, ubo, nir_imm_int(&b, index * 16), 8);
		} else if (eb == 32) {
			r = s8_load_ubo(&b, n, 32, ubo, nir_imm_int(&b, index * 16), 4);
		} else {
			/* 8/16/1-bit elements: unpack from 32-bit words */
			nir_def *w = s8_load_ubo(&b, 2, 32, ubo, nir_imm_int(&b, index * 16), 8);
			unsigned eb_store = eb == 1 ? 8 : eb;
			nir_def *c[16];
			for (unsigned i = 0; i < n; i++) {
				unsigned bit = i * eb_store;
				nir_def *word = nir_channel(&b, w, bit / 32);
				nir_def *e = nir_ubitfield_extract(&b, word, nir_imm_int(&b, bit % 32), nir_imm_int(&b, eb_store));
				c[i] = eb == 1 ? nir_ine_imm(&b, e, 0) : nir_u2uN(&b, e, eb);
			}
			r = nir_vec(&b, c, n);
		}
		vals[&ci] = { r };
		return;
	}
	if (auto *ii = dyn_cast<IntrinsicInst>(&ci)) {
		switch (ii->getIntrinsicID()) {
		case Intrinsic::fabs: vals[&ci] = { nir_fabs(&b, def(ii->getArgOperand(0))) }; return;
		case Intrinsic::bswap: {
			/* byte swap of a 32-bit value (VideoToolbox) */
			nir_def *x = def(ii->getArgOperand(0));
			if (x->bit_size != 32)
				break;
			nir_def *lo = nir_ior(&b, nir_ishl_imm(&b, nir_iand_imm(&b, x, 0xff), 24),
			    nir_ishl_imm(&b, nir_iand_imm(&b, x, 0xff00), 8));
			nir_def *hi = nir_ior(&b, nir_iand_imm(&b, nir_ushr_imm(&b, x, 8), 0xff00), nir_ushr_imm(&b, x, 24));
			vals[&ci] = { nir_ior(&b, lo, hi) };
			return;
		}
		case Intrinsic::sqrt: vals[&ci] = { nir_fsqrt(&b, def(ii->getArgOperand(0))) }; return;
		case Intrinsic::floor: vals[&ci] = { nir_ffloor(&b, def(ii->getArgOperand(0))) }; return;
		case Intrinsic::ceil: vals[&ci] = { nir_fceil(&b, def(ii->getArgOperand(0))) }; return;
		case Intrinsic::trunc: vals[&ci] = { nir_ftrunc(&b, def(ii->getArgOperand(0))) }; return;
		case Intrinsic::rint:
		case Intrinsic::roundeven: vals[&ci] = { nir_fround_even(&b, def(ii->getArgOperand(0))) }; return;
		case Intrinsic::exp2: vals[&ci] = { nir_fexp2(&b, def(ii->getArgOperand(0))) }; return;
		case Intrinsic::log2: vals[&ci] = { nir_flog2(&b, def(ii->getArgOperand(0))) }; return;
		case Intrinsic::minnum:
		case Intrinsic::minimum: vals[&ci] = { nir_fmin(&b, def(ii->getArgOperand(0)), def(ii->getArgOperand(1))) }; return;
		case Intrinsic::maxnum:
		case Intrinsic::maximum: vals[&ci] = { nir_fmax(&b, def(ii->getArgOperand(0)), def(ii->getArgOperand(1))) }; return;
		case Intrinsic::fma:
		case Intrinsic::fmuladd: vals[&ci] = { nir_ffma(&b, def(ii->getArgOperand(0)), def(ii->getArgOperand(1)), def(ii->getArgOperand(2))) }; return;
		case Intrinsic::copysign: {
			nir_def *x = def(ii->getArgOperand(0)), *y = def(ii->getArgOperand(1));
			vals[&ci] = { nir_bcsel(&b, nir_flt(&b, y, nir_imm_floatN_t(&b, 0, y->bit_size)),
			    nir_fneg(&b, nir_fabs(&b, x)), nir_fabs(&b, x)) };
			return;
		}
		case Intrinsic::smin: vals[&ci] = { nir_imin(&b, def(ii->getArgOperand(0)), def(ii->getArgOperand(1))) }; return;
		case Intrinsic::smax: vals[&ci] = { nir_imax(&b, def(ii->getArgOperand(0)), def(ii->getArgOperand(1))) }; return;
		case Intrinsic::umin: vals[&ci] = { nir_umin(&b, def(ii->getArgOperand(0)), def(ii->getArgOperand(1))) }; return;
		case Intrinsic::umax: vals[&ci] = { nir_umax(&b, def(ii->getArgOperand(0)), def(ii->getArgOperand(1))) }; return;
		case Intrinsic::abs: vals[&ci] = { nir_iabs(&b, def(ii->getArgOperand(0))) }; return;
		case Intrinsic::lifetime_start:
		case Intrinsic::lifetime_end:
		case Intrinsic::assume:
		case Intrinsic::experimental_noalias_scope_decl:
			return;
		case Intrinsic::vector_reduce_or:
			vals[&ci] = { nir_bany(&b, def(ii->getArgOperand(0))) };
			return;
		case Intrinsic::vector_reduce_and:
			vals[&ci] = { nir_ball(&b, def(ii->getArgOperand(0))) };
			return;
		case Intrinsic::vector_reduce_fadd: {
			nir_def *v = def(ii->getArgOperand(1));
			nir_def *s = def(ii->getArgOperand(0));
			for (unsigned i = 0; i < v->num_components; i++)
				s = nir_fadd(&b, s, nir_channel(&b, v, i));
			vals[&ci] = { s };
			return;
		}
		default:
			break;
		}
	}
	unsupported.push_back("call " + name.str());
	Type *t = ci.getType();
	if (!t->isVoidTy())
		vals[&ci] = { nir_undef(&b, isa<FixedVectorType>(t) ? cast<FixedVectorType>(t)->getNumElements() : 1, bits(t)) };
}

void Translator::emit(Instruction &i)
{
	nir_def *r = nullptr;
	Type *t = i.getType();
	if (auto *bo = dyn_cast<BinaryOperator>(&i)) {
		nir_def *x = def(bo->getOperand(0)), *y = def(bo->getOperand(1));
		switch (bo->getOpcode()) {
		case Instruction::Add: r = nir_iadd(&b, x, y); break;
		case Instruction::Sub: r = nir_isub(&b, x, y); break;
		case Instruction::Mul: r = nir_imul(&b, x, y); break;
		case Instruction::UDiv: r = nir_udiv(&b, x, y); break;
		case Instruction::SDiv: r = nir_idiv(&b, x, y); break;
		case Instruction::URem: r = nir_umod(&b, x, y); break;
		case Instruction::SRem: r = nir_irem(&b, x, y); break;
		case Instruction::Shl: r = nir_ishl(&b, x, nir_u2u32(&b, y)); break;
		case Instruction::LShr: r = nir_ushr(&b, x, nir_u2u32(&b, y)); break;
		case Instruction::AShr: r = nir_ishr(&b, x, nir_u2u32(&b, y)); break;
		case Instruction::And: r = x->bit_size == 1 ? nir_iand(&b, x, y) : nir_iand(&b, x, y); break;
		case Instruction::Or: r = nir_ior(&b, x, y); break;
		case Instruction::Xor: r = nir_ixor(&b, x, y); break;
		case Instruction::FAdd: r = nir_fadd(&b, x, y); break;
		case Instruction::FSub: r = nir_fsub(&b, x, y); break;
		case Instruction::FMul: r = nir_fmul(&b, x, y); break;
		case Instruction::FDiv: r = nir_fdiv(&b, x, y); break;
		case Instruction::FRem: r = nir_fmod(&b, x, y); break;
		default: fail("binary operator " + std::string(bo->getOpcodeName()));
		}
	} else if (auto *un = dyn_cast<UnaryOperator>(&i)) {
		r = nir_fneg(&b, def(un->getOperand(0)));
	} else if (auto *cmp = dyn_cast<FCmpInst>(&i)) {
		nir_def *x = def(cmp->getOperand(0)), *y = def(cmp->getOperand(1));
		switch (cmp->getPredicate()) {
		case CmpInst::FCMP_OEQ: case CmpInst::FCMP_UEQ: r = nir_feq(&b, x, y); break;
		case CmpInst::FCMP_ONE: case CmpInst::FCMP_UNE: r = nir_fneu(&b, x, y); break;
		case CmpInst::FCMP_OLT: case CmpInst::FCMP_ULT: r = nir_flt(&b, x, y); break;
		case CmpInst::FCMP_OLE: case CmpInst::FCMP_ULE: r = nir_fge(&b, y, x); break;
		case CmpInst::FCMP_OGT: case CmpInst::FCMP_UGT: r = nir_flt(&b, y, x); break;
		case CmpInst::FCMP_OGE: case CmpInst::FCMP_UGE: r = nir_fge(&b, x, y); break;
		case CmpInst::FCMP_TRUE: r = nir_imm_true(&b); break;
		case CmpInst::FCMP_FALSE: r = nir_imm_false(&b); break;
		case CmpInst::FCMP_ORD: r = nir_iand(&b, nir_feq(&b, x, x), nir_feq(&b, y, y)); break;
		case CmpInst::FCMP_UNO: r = nir_ior(&b, nir_fneu(&b, x, x), nir_fneu(&b, y, y)); break;
		default: fail("fcmp predicate");
		}
		if (x->num_components > 1 && r->num_components == 1 && !isa<FixedVectorType>(t))
			;
	} else if (auto *cmp = dyn_cast<ICmpInst>(&i)) {
		nir_def *x = def(cmp->getOperand(0)), *y = def(cmp->getOperand(1));
		switch (cmp->getPredicate()) {
		case CmpInst::ICMP_EQ: r = nir_ieq(&b, x, y); break;
		case CmpInst::ICMP_NE: r = nir_ine(&b, x, y); break;
		case CmpInst::ICMP_ULT: r = nir_ult(&b, x, y); break;
		case CmpInst::ICMP_ULE: r = nir_uge(&b, y, x); break;
		case CmpInst::ICMP_UGT: r = nir_ult(&b, y, x); break;
		case CmpInst::ICMP_UGE: r = nir_uge(&b, x, y); break;
		case CmpInst::ICMP_SLT: r = nir_ilt(&b, x, y); break;
		case CmpInst::ICMP_SLE: r = nir_ige(&b, y, x); break;
		case CmpInst::ICMP_SGT: r = nir_ilt(&b, y, x); break;
		case CmpInst::ICMP_SGE: r = nir_ige(&b, x, y); break;
		default: fail("icmp predicate");
		}
	} else if (auto *sel = dyn_cast<SelectInst>(&i)) {
		if (t->isStructTy()) {
			auto a = agg(sel->getTrueValue()), c = agg(sel->getFalseValue());
			nir_def *cond = def(sel->getCondition());
			std::vector<nir_def *> o;
			for (size_t k = 0; k < a.size(); k++)
				o.push_back(nir_bcsel(&b, cond, a[k], c[k]));
			vals[&i] = o;
			return;
		}
		r = nir_bcsel(&b, def(sel->getCondition()), def(sel->getTrueValue()), def(sel->getFalseValue()));
	} else if (auto *cst = dyn_cast<CastInst>(&i)) {
		nir_def *x = def(cst->getOperand(0));
		unsigned db = bits(t);
		switch (cst->getOpcode()) {
		case Instruction::Trunc: r = db == 1 ? nir_ine_imm(&b, nir_iand_imm(&b, x, 1), 0) : nir_u2uN(&b, x, db); break;
		case Instruction::ZExt: r = x->bit_size == 1 ? nir_b2iN(&b, x, db) : nir_u2uN(&b, x, db); break;
		case Instruction::SExt: r = x->bit_size == 1 ? nir_ineg(&b, nir_b2iN(&b, x, db)) : nir_i2iN(&b, x, db); break;
		case Instruction::FPTrunc:
		case Instruction::FPExt: r = nir_f2fN(&b, x, db); break;
		case Instruction::FPToUI: r = nir_f2uN(&b, x, db); break;
		case Instruction::FPToSI: r = nir_f2iN(&b, x, db); break;
		case Instruction::UIToFP: r = x->bit_size == 1 ? nir_b2fN(&b, x, db) : nir_u2fN(&b, x, db); break;
		case Instruction::SIToFP: r = nir_i2fN(&b, x, db); break;
		case Instruction::BitCast: {
			unsigned n = isa<FixedVectorType>(t) ? cast<FixedVectorType>(t)->getNumElements() : 1;
			r = nir_bitcast_vector(&b, x, db);
			(void)n;
			break;
		}
		case Instruction::AddrSpaceCast:
		case Instruction::PtrToInt:
		case Instruction::IntToPtr: r = x; break;
		default: fail("cast " + std::string(cst->getOpcodeName()));
		}
	} else if (auto *ee = dyn_cast<ExtractElementInst>(&i)) {
		nir_def *v = def(ee->getVectorOperand());
		if (auto *ci = dyn_cast<ConstantInt>(ee->getIndexOperand()))
			r = nir_channel(&b, v, ci->getZExtValue());
		else
			r = nir_vector_extract(&b, v, nir_u2u32(&b, def(ee->getIndexOperand())));
	} else if (auto *ie = dyn_cast<InsertElementInst>(&i)) {
		nir_def *v = def(ie->getOperand(0)), *e = def(ie->getOperand(1));
		if (auto *ci = dyn_cast<ConstantInt>(ie->getOperand(2)))
			r = nir_vector_insert_imm(&b, v, e, ci->getZExtValue());
		else
			r = nir_vector_insert(&b, v, e, nir_u2u32(&b, def(ie->getOperand(2))));
	} else if (auto *sv = dyn_cast<ShuffleVectorInst>(&i)) {
		nir_def *x = def(sv->getOperand(0)), *y = def(sv->getOperand(1));
		ArrayRef<int> mask = sv->getShuffleMask();
		nir_def *c[NIR_MAX_VEC_COMPONENTS];
		for (size_t k = 0; k < mask.size(); k++) {
			int m = mask[k];
			if (m < 0)
				c[k] = nir_undef(&b, 1, x->bit_size);
			else if ((unsigned)m < x->num_components)
				c[k] = nir_channel(&b, x, m);
			else
				c[k] = nir_channel(&b, y, m - x->num_components);
		}
		r = nir_vec(&b, c, mask.size());
	} else if (auto *ev = dyn_cast<ExtractValueInst>(&i)) {
		auto a = agg(ev->getAggregateOperand());
		r = a.at(ev->getIndices()[0]);
	} else if (auto *iv = dyn_cast<InsertValueInst>(&i)) {
		auto a = agg(iv->getAggregateOperand());
		a.at(iv->getIndices()[0]) = def(iv->getInsertedValueOperand());
		vals[&i] = a;
		return;
	} else if (auto *ld = dyn_cast<LoadInst>(&i)) {
		const Value *p = ld->getPointerOperand();
		unsigned n = isa<FixedVectorType>(t) ? cast<FixedVectorType>(t)->getNumElements() : 1;
		unsigned bs = bits(t);
		if (auto *gv = dyn_cast<GlobalVariable>(p->stripPointerCasts())) {
			if (gv->hasInitializer() && gv->isConstant()) {
				r = def(gv->getInitializer());
			} else {
				unsupported.push_back("load from global " + gv->getName().str());
				r = nir_undef(&b, n, bs);
			}
		} else if (isa<AllocaInst>(getUnderlyingObject(p))) {
			unsupported.push_back("load from alloca (dynamic indexing)");
			r = nir_undef(&b, n, bs);
		} else {
			nir_def *ptr = def(p);
			unsigned align = std::max((unsigned)ld->getAlign().value(), bs == 1 ? 1u : bs / 8);
			unsigned lbs = bs == 1 ? 8 : bs;
			if (ld->getPointerAddressSpace() == 2)
				r = s8_load_ubo(&b, n, lbs, nir_u2u32(&b, nir_ushr_imm(&b, ptr, 32)), nir_u2u32(&b, ptr), align);
			else
				r = nir_load_global(&b, ptr, align, n, lbs);
			if (bs == 1)
				r = nir_ine_imm(&b, r, 0);
		}
	} else if (auto *st = dyn_cast<StoreInst>(&i)) {
		unsupported.push_back("store");
		(void)st;
		return;
	} else if (isa<AllocaInst>(&i)) {
		return;
	} else if (auto *g = dyn_cast<GetElementPtrInst>(&i)) {
		r = gep(cast<GEPOperator>(g));
	} else if (auto *ci = dyn_cast<CallInst>(&i)) {
		emit_call(*ci);
		return;
	} else if (auto *br = dyn_cast<BranchInst>(&i)) {
		const BasicBlock *from = br->getParent();
		if (br->isUnconditional()) {
			branch_to(from, br->getSuccessor(0));
			nir_goto(&b, blocks[br->getSuccessor(0)]);
		} else {
			nir_def *cond = def(br->getCondition());
			/* phi stores must happen on the edge: route through edge blocks */
			auto edge = [&](const BasicBlock *to) {
				if (to->phis().empty())
					return blocks[to];
				nir_block *e = nir_block_create(b.shader);
				exec_list_push_tail(&b.impl->body, &e->cf_node.node);
				e->cf_node.parent = &b.impl->cf_node;
				nir_cursor save = b.cursor;
				b.cursor = nir_after_block(e);
				branch_to(from, to);
				nir_goto(&b, blocks[to]);
				b.cursor = save;
				return e;
			};
			nir_block *tb = edge(br->getSuccessor(0)), *fb = edge(br->getSuccessor(1));
			nir_goto_if(&b, tb, cond, fb);
		}
		return;
	} else if (auto *sw = dyn_cast<SwitchInst>(&i)) {
		nir_def *sel = def(sw->getCondition());
		const BasicBlock *from = sw->getParent();
		for (auto c : sw->cases()) {
			nir_block *next = nir_block_create(b.shader);
			exec_list_push_tail(&b.impl->body, &next->cf_node.node);
			next->cf_node.parent = &b.impl->cf_node;
			nir_block *target = nir_block_create(b.shader);
			exec_list_push_tail(&b.impl->body, &target->cf_node.node);
			target->cf_node.parent = &b.impl->cf_node;
			nir_goto_if(&b, target, nir_ieq_imm(&b, sel, c.getCaseValue()->getZExtValue()), next);
			b.cursor = nir_after_block(target);
			branch_to(from, c.getCaseSuccessor());
			nir_goto(&b, blocks[c.getCaseSuccessor()]);
			b.cursor = nir_after_block(next);
		}
		branch_to(from, sw->getDefaultDest());
		nir_goto(&b, blocks[sw->getDefaultDest()]);
		return;
	} else if (auto *ret = dyn_cast<ReturnInst>(&i)) {
		if (ret->getReturnValue())
			store_return(ret->getReturnValue());
		nir_goto(&b, b.impl->end_block);
		return;
	} else if (isa<UnreachableInst>(&i)) {
		nir_goto(&b, b.impl->end_block);
		return;
	} else if (isa<FreezeInst>(&i)) {
		r = def(i.getOperand(0));
	} else {
		fail("instruction " + std::string(i.getOpcodeName()));
	}
	if (r)
		vals[&i] = { r };
}

/* ---- driver-side lowering after Mesa's preprocessing ---- */

static bool lower_sysvals(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
	(void)data;
	unsigned off;
	switch (intr->intrinsic) {
	case nir_intrinsic_load_viewport_scale: off = 0; break;
	case nir_intrinsic_load_viewport_offset: off = 16; break;
	case nir_intrinsic_load_raw_vertex_offset_pan:
		/* first vertex (or lowest index) of the draw: vertex_id =
		 * raw_vertex_id + this (pan_lower_vertex_id) */
		b->cursor = nir_before_instr(&intr->instr);
		nir_def_replace(&intr->def, s8_load_ubo(b, 1, 32, nir_imm_int(b, S8_UBO_SYSVALS),
		    nir_imm_int(b, S8_SV_VERTEX_OFFSET), 4));
		return true;
	case nir_intrinsic_load_rt_conversion_pan: {
		/* framebuffer fetch: the driver stores the render targets' internal
		 * conversion words (pan_blend_get_internal_desc >> 32) at 32 +
		 * (rt * 2 + (16-bit ? 1 : 0)) * 4 */
		unsigned size = nir_alu_type_get_type_size(nir_intrinsic_src_type(intr));
		off = 32 + (nir_intrinsic_base(intr) * 2 + (size == 16 ? 1 : 0)) * 4;
		b->cursor = nir_before_instr(&intr->instr);
		nir_def_replace(&intr->def, s8_load_ubo(b, 1, 32, nir_imm_int(b, S8_UBO_SYSVALS), nir_imm_int(b, off), 4));
		return true;
	}
	default: return false;
	}
	b->cursor = nir_before_instr(&intr->instr);
	nir_def *v = s8_load_ubo(b, 3, 32, nir_imm_int(b, S8_UBO_SYSVALS), nir_imm_int(b, off), 16);
	nir_def_replace(&intr->def, v);
	return true;
}

/* Texture size queries: the driver writes each bound texture's width and
 * height (u32 each) into UBO S8_UBO_SYSVALS at S8_SV_TEX_SIZE + index * 8;
 * only level 0 is stored, so the lod just shifts. */
static bool lower_txs(nir_builder *b, nir_instr *instr, void *data)
{
	(void)data;
	if (instr->type != nir_instr_type_tex)
		return false;
	nir_tex_instr *tex = nir_instr_as_tex(instr);
	if (tex->op != nir_texop_txs && tex->op != nir_texop_query_levels)
		return false;
	b->cursor = nir_before_instr(instr);
	if (tex->op == nir_texop_query_levels) {
		nir_def_replace(&tex->def, nir_imm_int(b, 1));
		return true;
	}
	nir_def *index = nir_imm_int(b, tex->texture_index);
	int off = nir_tex_instr_src_index(tex, nir_tex_src_texture_offset);
	if (off >= 0)
		index = nir_iadd(b, index, nir_u2u32(b, tex->src[off].src.ssa));
	nir_def *size = s8_load_ubo(b, 2, 32, nir_imm_int(b, S8_UBO_SYSVALS),
	    nir_iadd_imm(b, nir_imul_imm(b, index, 8), S8_SV_TEX_SIZE), 8);
	int lod = nir_tex_instr_src_index(tex, nir_tex_src_lod);
	if (lod >= 0)
		size = nir_umax(b, nir_ushr(b, size, nir_u2u32(b, tex->src[lod].src.ssa)), nir_imm_int(b, 1));
	nir_def *res = size;
	unsigned n = tex->def.num_components;
	if (n == 1)
		res = nir_channel(b, size, 0);
	else if (n > 2)
		res = nir_vec3(b, nir_channel(b, size, 0), nir_channel(b, size, 1), nir_imm_int(b, 1));
	if (tex->def.bit_size != 32)
		res = nir_u2uN(b, res, tex->def.bit_size);
	nir_def_replace(&tex->def, res);
	return true;
}

static unsigned s8_lower16(const nir_instr *instr, void *data)
{
	(void)data;
	if (instr->type != nir_instr_type_alu)
		return 0;
	const nir_alu_instr *alu = nir_instr_as_alu(instr);
	/* conversions keep their sizes (they are the 8/16/32-bit boundary) */
	if (nir_op_infos[alu->op].is_conversion || nir_op_is_vec_or_mov(alu->op))
		return 0;
	/* 8-bit ALU too: Mesa vectorises it to v4i8 Bifrost instructions, and
	 * on the G71 a QuartzCore fragment shader faulted INSTR_INVALID_ENC
	 * inside its first 501 quadwords, where the only instruction the probe
	 * run had not proven is ICMP.v4i8.m1.ne (bool function constants). */
	if (alu->def.bit_size == 16 || alu->def.bit_size == 8)
		return 32;
	for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++)
		if (nir_src_bit_size(alu->src[i].src) == 16 || nir_src_bit_size(alu->src[i].src) == 8)
			return 32;
	return 0;
}

/* Frame shader that reloads render target 0 (MTLLoadActionLoad): texel
 * fetch of texture 0 at the pixel, as PanVK's fb preload. */
static nir_shader *preload_shader()
{
	nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, pan_shader_get_compiler_options(6),
	    "s8-preload");
	nir_def *coords = nir_u2u32(&b, nir_load_pixel_coord(&b));
	nir_tex_instr *tex = nir_tex_instr_create(b.shader, 2);
	tex->op = nir_texop_txf;
	tex->dest_type = nir_type_float32;
	tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
	tex->coord_components = 2;
	tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coords);
	tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_int(&b, 0));
	tex->texture_index = 0;
	tex->sampler_index = 0;
	nir_def_init(&tex->instr, &tex->def, 4, 32);
	nir_builder_instr_insert(&b, &tex->instr);
	nir_intrinsic_instr *st = nir_intrinsic_instr_create(b.shader, nir_intrinsic_store_output);
	st->num_components = 4;
	st->src[0] = nir_src_for_ssa(&tex->def);
	st->src[1] = nir_src_for_ssa(nir_imm_int(&b, 0));
	nir_intrinsic_set_base(st, 0);
	nir_intrinsic_set_write_mask(st, 0xf);
	nir_intrinsic_set_component(st, 0);
	nir_intrinsic_set_src_type(st, nir_type_float32);
	nir_io_semantics io = {};
	io.location = FRAG_RESULT_DATA0;
	io.num_slots = 1;
	nir_intrinsic_set_io_semantics(st, io);
	nir_builder_instr_insert(&b, &st->instr);
	b.shader->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_DATA0);
	b.shader->info.num_textures = 1;
	return b.shader;
}

/* ---- output ---- */

/* Blend register format of a render target output (pan_blend_type_from_nir). */
static uint32_t regfmt(nir_alu_type t)
{
	switch (t) {
	case nir_type_float16: return MALI_REGISTER_FILE_FORMAT_F16;
	case nir_type_float32: return MALI_REGISTER_FILE_FORMAT_F32;
	case nir_type_int32: return MALI_REGISTER_FILE_FORMAT_I32;
	case nir_type_uint32: return MALI_REGISTER_FILE_FORMAT_U32;
	case nir_type_int16: return MALI_REGISTER_FILE_FORMAT_I16;
	case nir_type_uint16: return MALI_REGISTER_FILE_FORMAT_U16;
	default: return 0;
	}
}

/* Self-test pair for the driver (tools/dreamlte/metal/test): "@test_vs"
 * takes float2 position (attribute 0) and float2 texcoord (attribute 1),
 * scales the position by constant buffer 0's first float and passes the
 * texcoord on as varying slot VAR0; "@test_fs" samples texture 0 with
 * sampler 0 there and multiplies by constant buffer 0's float4. They use
 * the same ABI as translated Metal shaders. */
static nir_variable *test_var(nir_shader *s, nir_variable_mode mode, const char *name, const glsl_type *t,
    int location)
{
	nir_variable *v = nir_variable_create(s, mode, t, name);
	v->data.location = location;
	/* vertex inputs: attribute table entry, as Translator::run */
	v->data.driver_location = mode == nir_var_shader_in && s->info.stage == MESA_SHADER_VERTEX ?
	    location - VERT_ATTRIB_GENERIC0 : location;
	return v;
}

/* mode: 0 "@test_vs", 1 "@test_fs", 2 "@test_fs16" (texture sampled and
 * written as half floats, like CoreAnimation's shaders), 3 "@test_fbf"
 * (framebuffer fetch: the colour already there, as half, times constant
 * buffer 0, so LD_TILE and its conversion are tested). The driver's
 * self-test (s8d_selftest) draws with them on the phone. */
static nir_shader *test_shader(int mode)
{
	bool vertex = mode == 0;
	nir_builder b = nir_builder_init_simple_shader(vertex ? MESA_SHADER_VERTEX : MESA_SHADER_FRAGMENT,
	    pan_shader_get_compiler_options(6), vertex ? "s8-test-vs" : "s8-test-fs");
	nir_def *k = s8_load_ubo(&b, 4, 32, nir_imm_int(&b, 0), nir_imm_int(&b, 0), 16);
	if (mode == 4) {
		/* "@test_imgw": constant buffer 0's float4 stored into image 1 at
		 * the fragment's pixel (image stores), and also its colour */
		nir_def *fc = nir_f2i32(&b, nir_load_frag_coord(&b));
		nir_def *coord = nir_vec4(&b, nir_channel(&b, fc, 0), nir_channel(&b, fc, 1), nir_imm_int(&b, 0),
		    nir_imm_int(&b, 0));
		s8_image_store(&b, nir_imm_int(&b, 1), coord, k, nir_type_float32);
		nir_store_var(&b, test_var(b.shader, nir_var_shader_out, "color", glsl_vec4_type(), FRAG_RESULT_DATA0), k, 0xf);
		return b.shader;
	}
	if (mode == 2 || mode == 3) {
		const glsl_type *h4 = glsl_vector_type(GLSL_TYPE_FLOAT16, 4);
		nir_def *v;
		if (mode == 3) {
			nir_variable *fb = test_var(b.shader, nir_var_shader_out, "fb", h4, FRAG_RESULT_DATA0);
			fb->data.fb_fetch_output = true;
			b.shader->info.outputs_read |= BITFIELD64_BIT(FRAG_RESULT_DATA0);
			b.shader->info.fs.uses_fbfetch_output = true;
			v = nir_load_var(&b, fb);
		} else {
			nir_variable *in = test_var(b.shader, nir_var_shader_in, "tc", glsl_vec_type(2), VARYING_SLOT_VAR0);
			in->data.interpolation = INTERP_MODE_SMOOTH;
			nir_tex_instr *tex = nir_tex_instr_create(b.shader, 1);
			tex->op = nir_texop_tex;
			tex->dest_type = nir_type_float16;
			tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
			tex->coord_components = 2;
			tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, nir_load_var(&b, in));
			tex->texture_index = 0;
			tex->sampler_index = 0;
			nir_def_init(&tex->instr, &tex->def, 4, 16);
			nir_builder_instr_insert(&b, &tex->instr);
			b.shader->info.num_textures = 1;
			v = &tex->def;
		}
		nir_store_var(&b, test_var(b.shader, nir_var_shader_out, "color", h4, FRAG_RESULT_DATA0),
		    nir_fmul(&b, v, nir_f2f16(&b, k)), 0xf);
		return b.shader;
	}
	if (vertex) {
		nir_def *pos = nir_load_var(&b, test_var(b.shader, nir_var_shader_in, "pos", glsl_vec_type(2),
		    VERT_ATTRIB_GENERIC0));
		nir_def *tc = nir_load_var(&b, test_var(b.shader, nir_var_shader_in, "tc", glsl_vec_type(2),
		    VERT_ATTRIB_GENERIC0 + 1));
		nir_def *p = nir_fmul(&b, pos, nir_channel(&b, k, 0));
		nir_store_var(&b, test_var(b.shader, nir_var_shader_out, "position", glsl_vec4_type(), VARYING_SLOT_POS),
		    nir_vec4(&b, nir_channel(&b, p, 0), nir_channel(&b, p, 1), nir_imm_float(&b, 0), nir_imm_float(&b, 1)), 0xf);
		nir_store_var(&b, test_var(b.shader, nir_var_shader_out, "tc", glsl_vec_type(2), VARYING_SLOT_VAR0), tc, 0x3);
	} else {
		nir_variable *in = test_var(b.shader, nir_var_shader_in, "tc", glsl_vec_type(2), VARYING_SLOT_VAR0);
		in->data.interpolation = INTERP_MODE_SMOOTH;
		nir_def *tc = nir_load_var(&b, in);
		nir_tex_instr *tex = nir_tex_instr_create(b.shader, 1);
		tex->op = nir_texop_tex;
		tex->dest_type = nir_type_float32;
		tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
		tex->coord_components = 2;
		tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, tc);
		tex->texture_index = 0;
		tex->sampler_index = 0;
		nir_def_init(&tex->instr, &tex->def, 4, 32);
		nir_builder_instr_insert(&b, &tex->instr);
		nir_store_var(&b, test_var(b.shader, nir_var_shader_out, "color", glsl_vec4_type(), FRAG_RESULT_DATA0),
		    nir_fmul(&b, &tex->def, k), 0xf);
		b.shader->info.num_textures = 1;
	}
	return b.shader;
}

/* air2nir output (little endian):
 *   "S8SH" u32 version=2
 *   u32 stage (0 vertex, 4 fragment), u32 binary size, u32 work_reg_count,
 *   u32 attribute_count, u32 ubo_count, u32 texture_count, u32 sampler_count,
 *   u32 varying_count, then varying_count x { u32 location, u32 pipe_format },
 *   u32 inline_sampler_count, inline_sampler_count x u64 Metal sampler state,
 *   u32 fs_flags (bit0 can_discard, bit1 reads_frag_coord, bit2 writes_depth,
 *   bit3 reads_fb), u32 rsd_words, rsd_words x u32 (renderer state, shader
 *   pointer 0), u32 shader_ptr_word (word index of the pointer in the RSD),
 *   u32 constant_count, constant_count x { u32 index, u32 bits, char name[48] }
 *   (function constants read from UBO S8_UBO_FC, -D), 8 x u32 render
 *   target register formats (fragment), u32 data_size, data_size bytes of
 *   constant tables (UBO S8_UBO_DATA), u32 tls_size (thread local storage
 *   per thread), u32 writes_point_size, binary bytes.
 * FUNCTION "@preload" builds the driver's render target reload shader. */
int main(int argc, char **argv)
{
	if (argc < 4) {
		fprintf(stderr, "usage: air2nir METALLIB FUNCTION OUT [-v name,name,...] [-c name=bits ...] "
		    "[-d] [-D]\n"
		    "  -v: varying table, entries name or name=slot; -D: function constants at run time;\n"
		    "  -D -F: the -c values folded into the -D program (specialisations)\n");
		return 2;
	}
	std::string lib = argv[1], fname = argv[2], out = argv[3];
	std::vector<std::string> varyings;
	std::vector<ConstantValue> consts;
	bool dump = false, dynamic = false;
	for (int i = 4; i < argc; i++) {
		std::string a = argv[i];
		if (a == "-v" && i + 1 < argc) {
			std::string s = argv[++i];
			size_t p = 0;
			while (p <= s.size()) {
				size_t q = s.find(',', p);
				if (q == std::string::npos)
					q = s.size();
				if (q > p)
					varyings.push_back(s.substr(p, q - p));
				p = q + 1;
			}
		} else if (a == "-c" && i + 1 < argc) {
			std::string s = argv[++i];
			size_t eq = s.find('=');
			consts.push_back({ s.substr(0, eq), strtoull(s.c_str() + eq + 1, nullptr, 0) });
		} else if (a == "-d") {
			dump = true;
		} else if (a == "-D") {
			dynamic = true;
		} else if (a == "-F") {
			fold_mode = true;
		}
	}
	glsl_type_singleton_init_or_ref();
	LLVMContext ctx;
	std::unique_ptr<Module> mod;
	std::vector<MetallibFunction> funcs;
	gl_shader_stage stage;
	nir_shader *s;
	std::vector<InlineSampler> inline_samplers;
	std::vector<uint8_t> constant_data;
	std::vector<std::string> unsupported;
	bool writes_images = false;
	if (fname == "@preload") {
		/* the driver's frame shader for MTLLoadActionLoad (lib = unused) */
		stage = MESA_SHADER_FRAGMENT;
		s = preload_shader();
	} else if (fname == "@test_vs" || fname == "@test_fs" || fname == "@test_fs16" || fname == "@test_fbf" || fname == "@test_imgw") {
		stage = fname == "@test_vs" ? MESA_SHADER_VERTEX : MESA_SHADER_FRAGMENT;
		s = test_shader(fname == "@test_vs" ? 0 : fname == "@test_fs" ? 1 : fname == "@test_fs16" ? 2 : fname == "@test_fbf" ? 3 : 4);
		writes_images = fname == "@test_imgw";
	} else {
		MetallibFunction *fn = nullptr;
		funcs = read_metallib(lib);
		for (auto &f : funcs)
			if (f.name == fname)
				fn = &f;
		if (!fn)
			fail("no function " + fname);
		auto buf = MemoryBuffer::getMemBuffer(StringRef((const char *)fn->bitcode.data(), fn->bitcode.size()), fname, false);
		auto parsed = parseBitcodeFile(buf->getMemBufferRef(), ctx);
		if (!parsed)
			fail("cannot parse bitcode: " + toString(parsed.takeError()));
		mod = std::move(*parsed);
		Module &m = *mod;
		Function *f = m.getFunction(fname);
		if (!f)
			fail("no entry point in module");
		specialise(m, *f, consts, dynamic);
		if (dump)
			f->print(errs());

		stage = fn->type == 0 ? MESA_SHADER_VERTEX : MESA_SHADER_FRAGMENT;
		Translator tr(m, *f, stage, varyings);
		s = tr.run();
		inline_samplers = tr.inline_samplers;
		constant_data = tr.constant_data;
		unsupported = tr.unsupported;
		writes_images = tr.writes_images;
		for (auto &u : unsupported)
			fprintf(stderr, "air2nir: unsupported: %s\n", u.c_str());
	}

	const unsigned gpu_id = 0x60a00001;
	s->info.next_stage = stage == MESA_SHADER_VERTEX ? MESA_SHADER_FRAGMENT : MESA_SHADER_NONE;
	NIR_PASS(_, s, nir_lower_vars_to_ssa);
	NIR_PASS(_, s, nir_opt_dce);
	nir_shader_gather_info(s, nir_shader_get_entrypoint(s));
	pan_shader_preprocess(s, gpu_id);
	/* CoreAnimation's varyings are all perspective-correct. */
	if (stage == MESA_SHADER_VERTEX)
		NIR_PASS(_, s, pan_nir_lower_static_noperspective, 0);
	NIR_PASS(_, s, nir_shader_intrinsics_pass, lower_sysvals, nir_metadata_control_flow, nullptr);
	NIR_PASS(_, s, nir_shader_instructions_pass, lower_txs, nir_metadata_control_flow, nullptr);
	/* Resource counts follow the highest index used. */
	unsigned ntex = 0, nsamp = 0, nubo = 0;
	nir_foreach_function_impl(impl, s) {
		nir_foreach_block(block, impl) {
			nir_foreach_instr(instr, block) {
				if (instr->type == nir_instr_type_tex) {
					nir_tex_instr *t = nir_instr_as_tex(instr);
					ntex = MAX2(ntex, t->texture_index + 1);
					if (nir_tex_instr_need_sampler(t))
						nsamp = MAX2(nsamp, t->sampler_index + 1);
				} else if (instr->type == nir_instr_type_intrinsic) {
					nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
					if (in->intrinsic == nir_intrinsic_load_ubo && nir_src_is_const(in->src[0]))
						nubo = MAX2(nubo, nir_src_as_uint(in->src[0]) + 1);
				}
			}
		}
	}
	s->info.num_textures = ntex;
	s->info.num_ubos = nubo;
	if (dump)
		nir_print_shader(s, stderr);

	struct pan_compile_inputs inputs = {};
	inputs.gpu_id = gpu_id;
	inputs.pushable_ubos = 0;
	inputs.no_idvs = true;
	struct util_dynarray binary;
	util_dynarray_init(&binary, nullptr);
	struct pan_shader_info info;
	/* 16-bit ALU runs as 32-bit: Bifrost packs some v2i16/v2f16 swizzles
	 * Mesa's back end cannot encode (bi_packer "swz0 < 4" on
	 * fixed_frag_lph_cpf, backboardd's main shader). */
	NIR_PASS(_, s, nir_lower_bit_size, s8_lower16, nullptr);
	NIR_PASS(_, s, nir_opt_algebraic);
	NIR_PASS(_, s, nir_opt_dce);
	pan_shader_compile(s, &inputs, &binary, &info);
	if (stage == MESA_SHADER_VERTEX) {
		/* Attribute L is table entry L, with holes; pan_shader.c counts
		 * the attributes read. The RSD needs the highest entry + 1, as
		 * PanVK patches it (panvk_vX_shader.c): std_vert1_lph reads
		 * 0, 1 and 3, and with a count of 3 its vertex job hung the G71. */
		uint64_t gen = (info.attributes_read & VERT_BIT_GENERIC_ALL) >> VERT_ATTRIB_GENERIC0;
		info.attribute_count = util_last_bit64(gen);
	}
	/* s8mtl_draw.c always binds the full 32-entry texture and sampler
	 * tables (inline samplers from 16, the null texture at 31). The G71
	 * range-checks against the RSD counts, which pan_shader.c takes from
	 * textures_used (empty here): a fragment shader sampling with inline
	 * sampler 16 and counts 0 faulted with ADDRESS_RANGE (0x5a). */
	info.texture_count = 32;
	/* Image stores (write_texture_2d): the driver binds the fragment
	 * stage's 32 textures as images in the attribute table, which the G71
	 * range-checks against the RSD's attribute count. */
	if (writes_images && stage == MESA_SHADER_FRAGMENT)
		info.attribute_count = 32;
	info.sampler_count = 32;

	fprintf(stderr, "air2nir: %s: %u bytes, %u work registers, %u attributes, %u UBOs, %u textures, "
	    "%u samplers, %u varyings\n", fname.c_str(), binary.size, info.work_reg_count, info.attribute_count,
	    info.ubo_count, info.texture_count, info.sampler_count,
	    stage == MESA_SHADER_VERTEX ? info.varyings.output_count : info.varyings.input_count);
	if (dump)
		pan_shader_disassemble(stderr, binary.data, binary.size, gpu_id, false);

	std::vector<uint32_t> w;
	auto put = [&](uint32_t v) { w.push_back(v); };
	put(0x48533853); /* "S8SH" */
	put(2);
	put(stage);
	put(binary.size);
	put(info.work_reg_count);
	put(info.attribute_count);
	put(info.ubo_count);
	put(info.texture_count);
	put(info.sampler_count);
	unsigned nv = stage == MESA_SHADER_VERTEX ? info.varyings.output_count : info.varyings.input_count;
	put(nv);
	for (unsigned i = 0; i < nv; i++) {
		const pan_shader_varying &v = stage == MESA_SHADER_VERTEX ? info.varyings.output[i] : info.varyings.input[i];
		put(v.location);
		put(v.format);
	}
	put(inline_samplers.size());
	for (auto &is : inline_samplers) {
		put((uint32_t)is.state);
		put((uint32_t)(is.state >> 32));
	}
	uint32_t flags = 0;
	if (stage == MESA_SHADER_FRAGMENT) {
		flags |= info.fs.can_discard ? 1 : 0;
		flags |= info.fs.reads_frag_coord ? 2 : 0;
		flags |= info.fs.writes_depth ? 4 : 0;
		flags |= info.fs.outputs_read ? 8 : 0;
		flags |= writes_images ? 16 : 0;
	}
	put(flags);
	/* RSD template: pack twice with different shader pointers to find the
	 * pointer's word. */
	struct mali_renderer_state_packed rsd0, rsd1;
	pan_pack(&rsd0, RENDERER_STATE, cfg) { pan_shader_prepare_rsd(&info, 0, &cfg); }
	pan_pack(&rsd1, RENDERER_STATE, cfg) { pan_shader_prepare_rsd(&info, 0x100000000ull, &cfg); }
	unsigned nw = sizeof(rsd0) / 4, ptr_word = ~0u;
	for (unsigned i = 0; i < nw; i++)
		if (rsd0.opaque[i] != rsd1.opaque[i])
			ptr_word = i - 1;       /* the pointer's high word changed */
	put(nw);
	for (unsigned i = 0; i < nw; i++)
		put(rsd0.opaque[i]);
	put(ptr_word);
	/* v2: run-time function constants (-D) */
	put(dynamic_constants.size());
	for (auto &dc : dynamic_constants) {
		put(dc.index);
		put(dc.bits);
		char name[48] = {};
		strncpy(name, dc.name.c_str(), sizeof(name) - 1);
		for (unsigned i = 0; i < 12; i++) {
			uint32_t v;
			memcpy(&v, name + i * 4, 4);
			put(v);
		}
	}
	/* v2: fragment render target register formats (fixed-function blend) */
	for (unsigned rt = 0; rt < 8; rt++)
		put(stage == MESA_SHADER_FRAGMENT ? regfmt(info.bifrost.blend[rt].type) : 0);
	/* v2: constant tables (UBO S8_UBO_DATA), padded to words */
	constant_data.resize((constant_data.size() + 3) & ~(size_t)3);
	put(constant_data.size());
	for (size_t i = 0; i < constant_data.size(); i += 4) {
		uint32_t v;
		memcpy(&v, &constant_data[i], 4);
		put(v);
	}
	put(info.tls_size);
	put(stage == MESA_SHADER_VERTEX ? info.vs.writes_point_size : 0);
	FILE *o = fopen(out.c_str(), "wb");
	if (!o)
		fail("cannot write " + out);
	fwrite(w.data(), 4, w.size(), o);
	fwrite(binary.data, 1, binary.size, o);
	fclose(o);
	return unsupported.empty() ? 0 : 3;
}
