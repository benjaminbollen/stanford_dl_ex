extern crate collenchyma as co;

use co::framework::IFramework;
use co::frameworks::Native;
use co::tensor::SharedTensor;
use co::memory::MemoryType;

fn write_to_memory(mem: &mut MemoryType, data: &[f64]) {
        if let &mut MemoryType::Native(ref mut mem) = mem {
            let mut mem_buffer = mem.as_mut_slice::<f64>();
            for (index, datum) in data.iter().enumerate() {
                mem_buffer[index] = *datum;
            }
        }
}

#[test]
fn it_creates_tensor() {
// allocate memory
    let native = Native::new();
    let cpu = native.new_device(native.hardwares()).unwrap();
    let mut x = SharedTensor::<f64>::new(&cpu, &(50, 50)).unwrap();
    let payload: Vec<f64> = ::std::iter::repeat(0f64).take(x.capacity()).collect::<Vec<f64>>();
    write_to_memory(x.get_mut(&cpu).unwrap(), &payload);
}
