use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use numpy::PyReadonlyArray1;
use std::collections::{BTreeSet, BTreeMap};
use std::fs::File;
use flate2::read::GzDecoder;
use tar::Archive;
use seq_io::fasta::{Reader, Record};
use ndarray::ArrayView1;
use anyhow::{Result, anyhow};
use pyo3::exceptions::PyRuntimeError;

/// A Python module implemented in Rust.
#[pymodule]
fn mutation_prediction_native(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_alignments, m)?)?;

    Ok(())
}

#[pyfunction]
fn load_alignments<'py>(_py: Python <'py>, msa_path: &str, ids: PyReadonlyArray1<i64>, dataset_name: &str) -> PyResult<BTreeMap<String, (String, BTreeSet<String>)>> {
    let ids = ids.as_array();
    load_alignments_internal(msa_path, ids, dataset_name).map_err(|err| PyRuntimeError::new_err(format!("{:?}", err)))
}

fn load_alignments_internal(msa_path: &str, ids: ArrayView1<i64>, dataset_name: &str) -> Result<BTreeMap<String, (String, BTreeSet<String>)>> {
    let mut query_ids = BTreeSet::new();
    for id in ids {
        query_ids.insert(format!("{}_{}", dataset_name, id));
    }

    let mut alignments = BTreeMap::new();
    let mut archive = Archive::new(GzDecoder::new(File::open(msa_path)?));
    for entry in archive.entries()? {
        let file = entry?;
        let path = file.header().path()?;
        if path.file_name().ok_or(anyhow!("Invalid filenames!"))?.to_str().ok_or(anyhow!("Invalid filenames!"))?.ends_with(".a3m") {
            let name = path
                .file_stem().ok_or(anyhow!("Invalid filenames!"))?
                .to_str().ok_or(anyhow!("Invalid filenames!"))?
                .to_string();
            let mut reader = Reader::new(file);
            let first = reader.next().ok_or(anyhow!("Empty alignments in archive!"))??;
            if query_ids.contains(first.id()?) {
                while let Some(result) = reader.next() {
                    let record = result?;
                    let id = record.id()?;
                    if !alignments.contains_key(id) {
                        let seq = String::from_utf8(record.owned_seq())?;
                        let mut found_ids = BTreeSet::new();
                        found_ids.insert(name.to_string());
                        alignments.insert(id.to_string(), (seq, found_ids));
                    } else {
                        alignments.get_mut(id).unwrap().1.insert(name.to_string());
                    }
                }
            }
        }
    }
    return Ok(alignments)
}
