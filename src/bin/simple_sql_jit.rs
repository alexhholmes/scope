use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::{AddressSpace, IntPredicate, OptimizationLevel};

#[repr(C)]
struct Row {
    id: i64,
    value: i64,
}

enum Column { Id, Value }

enum Predicate {
    Gt(Column, i64),
    Eq(Column, i64),
}

struct Query {
    select: Column,
    predicate: Predicate,
}

struct CodeGenerator<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
}

type QueryFunc = unsafe extern "C" fn(*const Row, i64) -> i64;

impl CodeGenerator<'_> {
    fn compile_query(&self, query: &Query) -> Option<JitFunction<'_, QueryFunc>> {
        // Signature: fn(*Row, i64) -> i64
        let i64_type = self.context.i64_type();
        let row_type = self.context.struct_type(&[i64_type.into(), i64_type.into()], false);
        let row_ptr = self.context.ptr_type(AddressSpace::default());

        // Create the function type.
        // Takes pointer to row array + length, return i64 (aggregated result)
        let fn_type = i64_type.fn_type(&[row_ptr.into(), i64_type.into()], false);
        let function = self.module.add_function("query", fn_type, None);

        // Loop setup (entry block)
        let entry = self.context.append_basic_block(function, "entry");
        let loop_header = self.context.append_basic_block(function, "loop");
        let loop_body = self.context.append_basic_block(function, "body");
        let loop_exit = self.context.append_basic_block(function, "exit");

        self.builder.position_at_end(entry);

        // Accumulator for results (the running sum)
        let result_ptr = self.builder.build_alloca(i64_type, "result").ok()?;
        self.builder.build_store(result_ptr, i64_type.const_int(0, false)).ok()?;

        // Loop counter (`i`)
        let i_ptr = self.builder.build_alloca(i64_type, "i").ok()?;
        self.builder.build_store(i_ptr, i64_type.const_int(0, false)).ok()?;

        self.builder.build_unconditional_branch(loop_header).ok()?;

        // Loop condition: `while i < len`
        self.builder.position_at_end(loop_header);

        let i = self.builder.build_load(i64_type, i_ptr, "i").ok()?.into_int_value();
        let len = function.get_nth_param(1)?.into_int_value();
        let cond = self.builder.build_int_compare(IntPredicate::ULT, i, len, "cond").ok()?;

        self.builder.build_conditional_branch(cond, loop_body, loop_exit).ok()?;

        // Load current row (loop body start)
        self.builder.position_at_end(loop_body);

        // Get pointer to rows[i], `GEP` -> "get element pointer" (LLVM array indexing)
        let rows = function.get_nth_param(0)?.into_pointer_value();
        let row_ptr = unsafe {
            self.builder.build_gep(row_type, rows, &[i], "row_ptr").ok()?
        };

        // Evaluate predicate
        // Load the column we're filtering on
        let field_idx = match query.predicate {
            Predicate::Gt(Column::Id, _) | Predicate::Eq(Column::Id, _) => 0,
            Predicate::Gt(Column::Value, _) | Predicate::Eq(Column::Value, _) => 1,
        };

        let field_ptr = self.builder.build_struct_gep(row_type, row_ptr, field_idx, "field").ok()?;
        let field_val = self.builder.build_load(i64_type, field_ptr, "val").ok()?.into_int_value();

        // Compare
        // Load row field and compare against constant
        let (pred, const_val) = match query.predicate {
            Predicate::Gt(_, c) => (IntPredicate::SGT, c),
            Predicate::Eq(_, c) => (IntPredicate::EQ, c),
        };

        let const_int = i64_type.const_int(const_val as u64, false);
        let matches = self.builder.build_int_compare(pred, field_val, const_int, "match").ok()?;

        // Conditional Accumulation
        // If predicate is `true`, extract SELECT column and add to accumulator
        let then_block = self.context.append_basic_block(function, "then");
        let continue_block = self.context.append_basic_block(function, "continue");

        self.builder.build_conditional_branch(matches, then_block, continue_block).ok()?;

        // If predicate matches, add SELECT column to result
        self.builder.position_at_end(then_block);

        let select_idx = match query.select {
            Column::Id => 0,
            Column::Value => 1,
        };

        let select_ptr = self.builder.build_struct_gep(row_type, row_ptr, select_idx, "select").ok()?;
        let select_val = self.builder.build_load(i64_type, select_ptr, "selected").ok()?.into_int_value();

        let current_result = self.builder.build_load(i64_type, result_ptr, "current").ok()?.into_int_value();
        let new_result = self.builder.build_int_add(current_result, select_val, "new_result").ok()?;
        self.builder.build_store(result_ptr, new_result).ok()?;

        self.builder.build_unconditional_branch(continue_block).ok()?;

        // Loop increment
        // `i++`, jump back to the loop condition
        self.builder.position_at_end(continue_block);

        let i_val = self.builder.build_load(i64_type, i_ptr, "i").ok()?.into_int_value();
        let next_i = self.builder.build_int_add(i_val, i64_type.const_int(1, false), "next").ok()?;
        self.builder.build_store(i_ptr, next_i).ok()?;

        self.builder.build_unconditional_branch(loop_header).ok()?;

        // Return result
        // Load accumulated result, then return it
        self.builder.position_at_end(loop_exit);

        let final_result = self.builder.build_load(i64_type, result_ptr, "result").ok()?.into_int_value();
        self.builder.build_return(Some(&final_result)).ok()?;

        unsafe { self.execution_engine.get_function("query").ok() }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::create();
    let module = context.create_module("jit");
    let execution_engine = module.create_jit_execution_engine(OptimizationLevel::Aggressive)?;
    let codegen = CodeGenerator {
        context: &context,
        module,
        builder: context.create_builder(),
        execution_engine,
    };

    let rows = vec![
        Row{ id: 1, value: 100 },
        Row{ id: 2, value: 200 },
        Row{ id: 3, value: 300 },
    ];

    let query = Query {
        select: Column::Value,
        predicate: Predicate::Gt(Column::Id, 1),
    };

    let compiled = codegen.compile_query(&query);

    let result = unsafe { compiled.unwrap().call(rows.as_ptr(), rows.len() as i64) };

    println!("SELECT SUM(value) FROM table WHERE id > 1");
    println!("Result: {}", result);

    Ok(())
}
