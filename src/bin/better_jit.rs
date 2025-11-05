use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::OptimizationLevel;

type ArithmeticFunc = unsafe extern "C" fn(i64, i64) -> i64;

struct CodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
}

impl CodeGen<'_> {
    fn jit_compile_arithmetic_func(&self) -> Option<JitFunction<'_, ArithmeticFunc>> {
        let i64_type = self.context.i64_type();
        let fn_type = i64_type.fn_type(&[i64_type.into(), i64_type.into()], false);
        let function = self.module.add_function("arithmetic", fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");

        self.builder.position_at_end(basic_block);

        let x = function.get_nth_param(0)?.into_int_value();
        let y = function.get_nth_param(1)?.into_int_value();

        let arithmetic = self.builder.build_int_mul(x, y, "arithmetic").ok()?;
        let arithmetic = self.builder.build_int_add(x, arithmetic, "arithmetic").ok()?;

        self.builder.build_return(Some(&arithmetic)).unwrap();

        unsafe { self.execution_engine.get_function("arithmetic").ok() }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::create();
    let module = context.create_module("jit");
    let execution_engine = module.create_jit_execution_engine(OptimizationLevel::None)?;
    let codegen = CodeGen {
        context: &context,
        module,
        builder: context.create_builder(),
        execution_engine,
    };

    let arithmetic = codegen.jit_compile_arithmetic_func().ok_or("failed to JIT compile `arithmetic`")?;

    let x = 2i64;
    let y = 4i64;

    unsafe {
        println!("{}", arithmetic.call(x, y));
        assert_eq!(arithmetic.call(x, y), x * y + x)
    }

    Ok(())
}