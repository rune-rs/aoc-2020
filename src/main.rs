use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use notify::{DebouncedEvent, RecommendedWatcher, RecursiveMode, Watcher};
use rune::alloc::borrow::TryToOwned;
use rune::alloc::clone::TryClone;
use rune::compile::{meta, CompileVisitor, FileSourceLoader, MetaError, MetaRef};
use rune::runtime::{Function, VmResult};
use rune::termcolor::{ColorChoice, StandardStream};
use rune::{vm_try, Any, Context, Hash, Module, Source, Unit, Vm};
use rune::{Diagnostics, ItemBuf, Options, Sources};
use structopt::StructOpt;
use tokio::runtime;

struct FlatIter {
    loop_starts: Vec<i32>,
    loop_ends: Vec<i32>,
    loop_counters: Vec<i32>,
}

fn flat_loop<'a>(loop_starts: Vec<i32>, loop_ends: Vec<i32>) -> FlatIter {
    let mut loop_counters = loop_starts.clone();
    loop_counters[0] -= 1;
    FlatIter {
        loop_starts,
        loop_ends,
        loop_counters,
    }
}

impl FlatIter {
    fn next<'a>(&'a mut self) -> Option<&'a Vec<i32>> {
        let mut inc = 1;
        for (idx, counter) in self.loop_counters.iter_mut().enumerate() {
            *counter += inc;
            if *counter >= self.loop_ends[idx] {
                *counter = self.loop_starts[idx];
                inc = 1;
            } else {
                inc = 0;
            }
        }
        if inc == 1 {
            return None;
        }

        Some(&self.loop_counters)
    }
}

#[derive(Any)]
#[rune(item = ::aoc)]
struct CA3 {
    current_min: Vec<i32>,
    current_max: Vec<i32>,
    current_size: Vec<usize>,
    eval_cell_fn: Function,
    state: Vec<(Vec<i32>, bool)>,
    query: Vec<bool>,
}

impl CA3 {
    #[rune::function(path = Self::new)]
    fn new(dims: usize, eval_cell_fn: Function) -> Self {
        Self {
            current_min: vec![i32::MAX; dims],
            current_max: vec![i32::MIN; dims],
            current_size: vec![0; dims],
            eval_cell_fn,
            state: vec![],
            query: vec![],
        }
    }

    #[rune::function(keep)]
    fn add_slot(&mut self, position: Vec<i32>, value: bool) {
        self.current_min = position
            .iter()
            .zip(self.current_min.iter())
            .map(|(a, b)| i32::min(*a, *b))
            .collect();

        self.current_max = position
            .iter()
            .zip(self.current_max.iter())
            .map(|(a, b)| i32::max(*a + 1, *b))
            .collect();

        self.state.push((position, value));
    }

    fn build_query_structure(&mut self) {
        self.query.clear();

        let query_sizes = self
            .current_max
            .iter()
            .zip(self.current_min.iter())
            .map(|(a, b)| (*a - *b) as usize)
            .collect::<Vec<_>>();

        let query_size = query_sizes.iter().product();

        self.query.resize(query_size, false);
        self.current_size = query_sizes;

        for (pos, val) in &self.state {
            let idx = self.position_to_index(pos);
            self.query[idx] = *val;
        }
    }

    #[rune::function]
    fn render(&mut self, mut template: Vec<i32>) -> String {
        self.build_query_structure();

        let mut out = vec![];
        for y in self.current_min[1]..self.current_max[1] {
            let mut line = String::with_capacity(self.current_size[0]);
            for x in self.current_min[0]..self.current_max[0] {
                template[0] = x;
                template[1] = y;
                let idx = self.position_to_index(&template);
                line.push(if self.query[idx] { '#' } else { '.' })
            }
            out.push(line)
        }

        out.join("\n")
    }

    fn position_to_index(&self, pos: &Vec<i32>) -> usize {
        let relative_pos = pos
            .iter()
            .zip(self.current_min.iter())
            .map(|(&a, &b)| (a - b) as usize);

        let mut idx = 0;
        let mut scale = 1;
        for (it, p) in relative_pos.enumerate() {
            idx += p * scale;
            scale *= self.current_size[it];
        }

        idx
    }

    fn prepare_scratch(&mut self, position: &Vec<i32>) -> Vec<bool> {
        let mut scratch = vec![];

        let loop_starts = position.iter().map(|v| v - 1).collect::<Vec<_>>();
        let loop_ends = position.iter().map(|v| v + 2).collect::<Vec<_>>();
        let mut iter = flat_loop(loop_starts, loop_ends);
        while let Some(counter) = iter.next() {
            let mut is_ok = true;
            for (idx, loop_v) in counter.iter().enumerate() {
                if *loop_v < self.current_min[idx] || *loop_v >= self.current_max[idx] {
                    is_ok = false;
                    break;
                }
            }

            if is_ok && counter != position {
                let idx = self.position_to_index(counter);
                scratch.push(self.query[idx as usize]);
            }
        }
        scratch
    }

    #[rune::function]
    fn step(&mut self) -> VmResult<()> {
        self.build_query_structure();
        self.state.clear();
        let mut updates = Vec::with_capacity(self.query.len());

        let loop_starts = self.current_min.iter().map(|v| v - 1).collect::<Vec<_>>();
        let loop_ends = self.current_max.iter().map(|v| v + 1).collect::<Vec<_>>();
        let mut iter = flat_loop(loop_starts, loop_ends);

        while let Some(position) = iter.next() {
            let mut is_ok_index = true;
            for (idx, loop_v) in position.iter().enumerate() {
                if *loop_v < self.current_min[idx] || *loop_v >= self.current_max[idx] {
                    is_ok_index = false;
                    break;
                }
            }

            let state_here = if is_ok_index {
                let idx = self.position_to_index(&position);
                self.query[idx]
            } else {
                false
            };

            let scratch = self.prepare_scratch(&position);

            let res = vm_try!(self.eval_cell_fn.call::<bool>((state_here, scratch)));

            if res {
                updates.push((position.clone(), true));
            }
        }

        updates.drain(..).for_each(|v| self.add_slot(v.0, v.1));
        VmResult::Ok(())
    }

    #[rune::function]
    fn get_state(&mut self) -> Vec<bool> {
        self.build_query_structure();
        self.query.clone()
    }
}

fn ca_module() -> Result<Module, rune::ContextError> {
    let mut module = Module::with_crate("aoc")?;
    module.ty::<CA3>()?;
    module.function_meta(CA3::step)?;
    module.function_meta(CA3::get_state)?;
    module.function_meta(CA3::add_slot__meta)?;
    module.function_meta(CA3::render)?;
    module.function_meta(CA3::new)?;

    Ok(module)
}

pub struct ScriptEngineBuilder {
    root: PathBuf,
    modules: Vec<Module>,
}

impl ScriptEngineBuilder {
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            modules: vec![],
        }
    }

    pub fn add_module(mut self, module: Module) -> Self {
        self.modules.push(module);
        self
    }

    pub fn build(self) -> Result<ScriptEngine> {
        let rt = runtime::Builder::new_multi_thread().enable_all().build()?;
        let mut context = rune_modules::with_config(true)?;

        for module in self.modules {
            context.install(&module)?;
        }

        let context = Arc::new(context);
        let (tx, rx) = channel();

        let mut watcher: RecommendedWatcher = Watcher::new(tx, Duration::from_secs(2))?;
        watcher.watch(self.root.parent().unwrap(), RecursiveMode::Recursive)?;
        let unit: Arc<Unit> = Default::default();
        let mut engine = ScriptEngine {
            main_file: self.root,
            _watcher: watcher,
            rx,
            context,
            unit,
            sources: Sources::new(),
            rt,
            days: Default::default(),
            tests: Default::default(),
        };

        let _ = engine.reload();
        Ok(engine)
    }
}

#[derive(Default)]
pub struct TestVisitor {
    test_functions: Vec<(Hash, ItemBuf)>,
}

impl TestVisitor {
    /// Convert visitor into test functions.
    pub(crate) fn into_test_functions(self) -> Vec<(Hash, ItemBuf)> {
        self.test_functions
    }
}

impl CompileVisitor for TestVisitor {
    fn register_meta(&mut self, meta: MetaRef<'_>) -> Result<(), MetaError> {
        let type_hash = match &meta.kind {
            meta::Kind::Function { is_test, .. } if *is_test => meta.hash,
            _ => return Ok(()),
        };

        self.test_functions
            .push((type_hash, meta.item.try_to_owned()?));
        Ok(())
    }
}

pub struct ScriptEngine {
    _watcher: RecommendedWatcher,
    rx: Receiver<DebouncedEvent>,
    context: Arc<Context>,
    unit: Arc<Unit>,
    main_file: PathBuf,
    sources: Sources,
    rt: runtime::Runtime,
    days: HashMap<u32, f64>,
    tests: Vec<(Hash, ItemBuf)>,
}

impl ScriptEngine {
    fn reload(&mut self) -> Result<()> {
        let mut sources = Sources::new();
        sources.insert(Source::from_path(&self.main_file)?)?;

        let mut diagnostics = Diagnostics::without_warnings();
        let mut options = Options::default();
        options.debug_info(true);
        options.memoize_instance_fn(true);

        self.sources = sources;

        let mut test_finder = TestVisitor::default();
        let mut source_loader = FileSourceLoader::new();

        let result = rune::prepare(&mut self.sources)
            .with_context(&self.context)
            .with_options(&options)
            .with_diagnostics(&mut diagnostics)
            .with_visitor(&mut test_finder)?
            .with_source_loader(&mut source_loader)
            .build();

        if !diagnostics.is_empty() {
            let mut writer = StandardStream::stderr(ColorChoice::Always);
            diagnostics.emit(&mut writer, &self.sources)?;
        }

        let unit = result?;

        self.unit = Arc::new(unit);
        self.tests = test_finder.into_test_functions();

        for day in 0..24 {
            self.days.entry(day).or_default();
        }

        Ok(())
    }

    pub fn run_tests(&mut self) -> Result<bool> {
        // TODO: use rune-tests capture_output to stop prints from tests from showing
        let runtime = Arc::new(self.context.runtime()?);

        let start = Instant::now();
        let mut failures = HashMap::new();

        for test in &self.tests {
            let mut vm = Vm::new(runtime.clone(), self.unit.clone());
            let mut execution = vm.execute(test.0, ())?;

            match self.rt.block_on(execution.async_complete()).into_result() {
                Err(e) => {
                    // TODO: store output here
                    failures.insert(test.1.try_clone()?, e);
                    print!("F");
                }
                Ok(_) => {
                    print!(".")
                }
            }
        }

        println!("");

        let elapsed = start.elapsed();

        for (item, error) in &failures {
            println!("----------------------------------------");
            println!("Test: {}\n", item);

            let mut writer = StandardStream::stderr(ColorChoice::Always);
            error
                .emit(&mut writer, &self.sources)
                .expect("failed writing info");
        }

        println!("====");
        println!(
            "Ran {} tests with {} failures in {:.3} seconds",
            self.tests.len(),
            failures.len(),
            elapsed.as_secs_f64()
        );

        Ok(failures.is_empty())
    }

    pub fn call_all(&mut self) -> Result<()> {
        self.call_days(self.days.keys().copied().collect())
    }

    pub fn call_days(&mut self, mut days: Vec<u32>) -> Result<()> {
        days.sort_unstable();
        for day in days {
            self.run_day(day, 10)?;
        }
        Ok(())
    }

    pub fn run_day(&mut self, day: u32, iterations: u32) -> Result<f64> {
        let start = Instant::now();
        let func = &[&format!("day{}", day), "run"];
        let mut idx = 0;
        let runtime = Arc::new(self.context.runtime()?);
        let result = loop {
            let mut vm = Vm::new(runtime.clone(), self.unit.clone());
            let mut execution = vm.execute(func, ())?;

            let result = match self.rt.block_on(execution.async_complete()).into_result() {
                Err(e) => {
                    let mut writer = StandardStream::stderr(ColorChoice::Always);
                    e.emit(&mut writer, &self.sources)
                        .expect("failed writing info");
                    println!("Day {:>2}     FAILED", day);
                    return Err(e.into());
                }
                Ok(v) => v,
            };
            idx += 1;
            if idx == iterations {
                break result;
            }
        };

        let elapsed = start.elapsed();

        let previous_time = self.days[&day];
        let elapsed_ms = (elapsed.as_secs_f64() / iterations as f64) * 1000.0;
        let frac = 1.0 - elapsed_ms / previous_time;
        if frac.abs() > 0.05 && frac.is_finite() {
            let sign = if frac > 0.0 { '-' } else { '+' };
            println!(
                "Day {:>2}   {: >8.4}   {}{: >6.2}%     {:?}",
                day,
                elapsed_ms,
                sign,
                frac.abs() * 100.0,
                result,
            );
        } else {
            println!(
                "Day {:>2}   {: >8.5}   --------     {:?}",
                day, elapsed_ms, result,
            );
        }

        self.days.insert(day, elapsed_ms);
        Ok(elapsed_ms)
    }

    pub fn update(&mut self) -> Result<Vec<String>> {
        let updates = self
            .rx
            .try_iter()
            .filter_map(|v| match v {
                DebouncedEvent::Write(v) => {
                    Some(v.file_name().unwrap().to_str().unwrap().to_owned())
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        if !updates.is_empty() {
            self.reload().map(|_| updates)
        } else {
            Ok(updates)
        }
    }

    pub fn days(&self) -> Vec<u32> {
        self.days.keys().copied().collect()
    }
}

#[derive(StructOpt)]
struct AOC {
    #[structopt(subcommand)]
    cmd: Option<Command>,
}

#[derive(StructOpt)]
enum Command {
    Bench {
        #[structopt(short)]
        iterations: u32,

        #[structopt(short)]
        day: Option<u32>,
    },
    RunAll,
}

fn run_reload(mut engine: ScriptEngine) -> Result<()> {
    loop {
        let updated_files = match engine.update() {
            Ok(v) => v,
            Err(_) => {
                eprintln!("failed reloading, not running");
                continue;
            }
        };

        if updated_files.is_empty() {
            continue;
        }

        eprintln!("Starting hot reload...");
        if !engine.run_tests().unwrap_or(false) {
            eprintln!("failed tests; not rerunning...");
            continue;
        }

        let days = updated_files
            .iter()
            .filter(|v| v.starts_with("day"))
            .map(|v| v[3..].strip_suffix(".rn").unwrap().parse().expect("x"))
            .collect::<Vec<_>>();

        let non_days = updated_files
            .iter()
            .filter(|v| !v.starts_with("day"))
            .collect::<Vec<_>>();

        if !non_days.is_empty() {
            let _ = engine.call_all();
        } else {
            let _ = engine.call_days(days);
        }

        std::thread::sleep(Duration::from_millis(30));
    }
}

fn create_engine() -> Result<ScriptEngine> {
    ScriptEngineBuilder::new("script/main.rn".into())
        .add_module(ca_module()?)
        .build()
}
fn run(run_once: bool) -> Result<()> {
    let mut engine = create_engine()?;

    if !engine.run_tests().unwrap_or(false) {
        eprintln!("failed tests; not rerunning...");
    }

    if run_once {
        engine.call_all()
    } else {
        let _ = engine.call_all();
        run_reload(engine)
    }
}

fn bench(iterations: u32, day: Option<u32>) -> Result<()> {
    let mut engine = create_engine()?;

    if let Some(day) = day {
        println!("Benchmarking day {}, iterations={}", day, iterations);
        engine.run_day(day, iterations)?;
    } else {
        println!("Benchmarking all days, iterations={}", iterations);
        let mut days = engine.days();
        days.sort_unstable();
        for day in days {
            engine.run_day(day, iterations)?;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = AOC::from_args();

    if let Some(cmd) = args.cmd {
        match cmd {
            Command::Bench { iterations, day } => bench(iterations, day),
            Command::RunAll => run(true),
        }
    } else {
        run(false)
    }
}
