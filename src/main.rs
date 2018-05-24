#[macro_use]
extern crate ndarray as nd;
extern crate threadpool;

use std::env;
use std::io::Read;
use std::fs::File;
use std::sync::mpsc::channel;
use nd::{Array1, Array2, ArrayView1, ArrayView2, Dim};

static STEADY_STEPS : u32 = 25;
static NUM_STATE : usize = 61;
static NUM_INPUT : usize = 41;

// Accepts a filename and a dimension, and outputs a 2D ndarray
fn read_csv(filename : &String, dim : Dim<[usize; 2]>) -> Array2<f32> {
    println!("File path: {:?}", *filename);

    // Read file into string
    let mut file = File::open(filename).unwrap();
    let mut buffer = String::new();
    let _res = file.read_to_string(&mut buffer);

    // Split string into array of floats
    let mut str_mat: Vec<&str> = buffer.split(|c| (c=='\n') || (c==',')).collect();
    if str_mat.last().cloned().expect("") == "" { // remove last blank line
        str_mat.pop();
    }
    let f32_mat: Vec<f32> = str_mat.iter().map(|c| c.parse::<f32>().unwrap()).collect();
    Array2::from_shape_vec(dim, f32_mat).unwrap()
}

// Takes a mutable reference to an array, and applies elementwise sigmoid
fn squash(v : &Array1<f32>) -> Array1<f32> {
    let sq_clos = |x: f32| -> f32 { 1. / ((-1. * x).exp() + 1.) };
    v.mapv(sq_clos)
}

// Applies RNN to drug input and network 
fn rnn_forward(m : &Array2<f32>, i : &Array1<f32>, its : u32) -> Array1<f32> {
    let mut state = Array1::zeros(NUM_STATE);
    let i2s : ArrayView2<f32> = m.slice(s![.., ..NUM_INPUT]);
    let s2s : ArrayView2<f32> = m.slice(s![.., NUM_INPUT..]);
    let icontrib : Array1<f32> = i2s.dot(i);
    for _iter in 0..its {
        let lin = s2s.dot(&state) + &icontrib;
        state = squash(&lin);
    }
    state
}

fn main() {

    // receive command line arguments
    let cmd : Vec<String> = env::args().collect();

    let weight_mat_dim = Dim([NUM_STATE, NUM_INPUT + NUM_STATE]);
    let weight_mat : Array2<f32> = read_csv(&cmd[1], weight_mat_dim);
    println!("weight matrix from {}: {:?}", &cmd[1], weight_mat.shape());

    let mut ssri : Array1<f32> = Array1::zeros(NUM_INPUT);
    ssri[0] = 1.;
    ssri[40] = 1.;
    println!("drug input sized {:?}: {}", ssri.shape(), &ssri);
    let ss : Array1<f32> = rnn_forward(&weight_mat, &ssri, STEADY_STEPS);
    println!("ss sized {:?}: {}", ss.shape(), &ss);
    let amines : ArrayView1<f32> = ss.slice(s![23..26]);
    println!("amines {}", &amines);
}
