#[macro_use]
extern crate ndarray as nd;
extern crate threadpool;

use std::env;
use std::io::Read;
use std::fs::File;
use std::sync::mpsc::channel;
use nd::{Array1, Array2, ArrayView1, Dim};

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

// Applies one RNN iteration to a 1D ndarray
fn rnn_step(m : &Array2<f32>, v : &Array1<f32>) -> Array1<f32> {
    let lin = m.dot(v);
    squash(&lin)
}
fn rnn_forward(m : &Array2<f32>, its : u32) -> Array1<f32> {
    // TODO: next
}

fn main() {

    // receive command line arguments
    let cmd : Vec<String> = env::args().collect();

    let weight_mat_dim = Dim([NUM_STATE, NUM_INPUT + NUM_STATE]);
    let weight_mat : Array2<f32> = read_csv(&cmd[1], weight_mat_dim);
    println!("weight matrix: {}", weight_mat);

    let a = Array2::from_elem((5, 4), 3.);
    let b = Array1::from_elem(4, 2.);
    let b = squash(&b);
    let c : Array1<f32> = a.dot(&b);
    let c = squash(&c);
    let bcopy = rnn_step(&a, &b);
    assert!(bcopy == c);
    let d = Array2::from_shape_vec((2, 2), vec![1., 2., 3., 4.]).unwrap();
    println!("{:?}, {:?}", a, d);
    let e : ArrayView1<f32> = d.slice(s![0, ..]);
    let f : ArrayView1<f32> = d.slice(s![.., 1]);

    println!("a is ok: {:?}", a.shape());
    println!("a: {}", a);
    println!("{}, {}", a, b);
    println!("Dot product: {}", c);
    println!("Slice [0, ..]: {}", e);
    println!("Slice [.., 1]: {}", f);
}
