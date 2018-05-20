#[macro_use]
extern crate ndarray as nd;
extern crate threadpool;

use std::env;
use std::io::Read;
use std::fs::File;
use std::sync::mpsc::channel;
use nd::{Array1, Array2, ArrayView1};

//let squash  |x: f32| -> f32 { 1 / (powf(-1. * x) + 1) };

fn read_csv() -> () {
    let file_path = env::args_os().nth(1).unwrap();
    println!("File path: {:?}", file_path);

    // Read file into string
    let mut file = File::open(file_path).unwrap();
    let mut buffer = String::new();
    file.read_to_string(&mut buffer);

    // Split string into array of floats
    let mut str_rows: Vec<&str> = buffer.split('\n').collect();
    if str_rows.last().cloned().expect("") == "" {
        str_rows.pop();
    }
    let str_mat: Vec<Vec<&str>> = str_rows.iter().map(|r| r.split(',').collect()).collect();
    //let f32_mat: Vec<Vec<Result<f32,<f32 as std::str::FromStr>::Err>>> = str_mat.iter().map(|r| r.iter().map(|c| c.parse::<f32>()).collect()).collect();
    let f32_mat: Vec<Vec<f32>> = str_mat.iter().map(|r| r.iter().map(|c| c.parse::<f32>().unwrap()).collect()).collect();
    
    for row in f32_mat {
        println!(" |  {:?}", row);
    }
}

fn main() {
    read_csv();
    let b = Array1::from_elem(4, 2.);
    let mut a = Array2::from_elem((5, 4), 3.);
    a.mapv_inplace(|x| x + 1.);
    let c : Array1<f32> = a.dot(&b);
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
