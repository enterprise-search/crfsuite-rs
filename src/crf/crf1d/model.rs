use std::{convert::TryInto, ffi::CStr, fs::File, io::Write, path::PathBuf};

use cqdb::CQDB;
use crfsuite_sys::crfsuite_dictionary_t;
use libc::c_char;
use serde::{Deserialize, Serialize};

use crate::{crf::{model::Model, trainer::FeatType}, quark::{Quark, StringTable}};

use super::tagger::Crf1dTagger;

pub type FeatRefs = Vec<usize>;

#[repr(C)]
#[derive(Debug, Serialize, Deserialize)]
pub struct Feature {
    pub cat: FeatType,
    pub src: u32,
    pub dst: u32,
    pub weight: f64,
}

#[repr(C)]
pub struct feature_refs {
    num_features: i32,
    fids: *const i32,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct T {
    num_features: usize,
    features: Vec<Feature>,
    labels: Vec<String>,
    attrs: Vec<String>,
    attr_refs: Vec<FeatRefs>,
    label_refs: Vec<FeatRefs>,
}

const FT_STATE: u32 = 0;

#[no_mangle]
pub unsafe extern "C" fn save_model_r(
    fpath: *const c_char,
    w: *const f64,
    attrs: *const crfsuite_dictionary_t,
    labels: *const crfsuite_dictionary_t,
    L: usize,
    A: usize,
    K: usize,
    features: *const Feature,
    attr_refs: *const feature_refs,
    label_refs: *const feature_refs,
) {
    log::info!("writing model: {fpath:?}");
    let mut fmap: Vec<i32> = vec![-1; K];
    let mut amap: Vec<i32> = vec![-1; A];

    let mut J = 0;
    let mut B = 0;
    let mut t = T {
        num_features: K,
        ..Default::default()
    };
    for k in 0..K {
        let pw = w.offset(k as isize);
        if *pw != 0.0 {
            fmap[k] = J;
            J += 1;
            let f = features
                .offset(k as isize)
                .as_ref()
                .expect("failed to read feature");
            let mut src = f.src;
            if f.cat == FeatType::FT_STATE {
                if amap[f.src as usize] < 0 {
                    amap[f.src as usize] = B;
                    B += 1;
                    src = amap[f.src as usize] as u32;
                }
            }
            let f = Feature {
                cat: f.cat,
                src: src,
                dst: f.dst,
                weight: *pw,
            };
            t.features.push(f);
        }
    }
    log::info!("features: {J}/{K}, attrs: {B}/{A}, labels: {L}");
    for l in 0..L {
        let mut label: *mut libc::c_char = std::ptr::null_mut();
        let ret = (*labels)
            .to_string
            .map(|f| {
                f(
                    labels as *mut _,
                    l as i32,
                    &mut label as *mut *mut _ as *mut *const _,
                )
            })
            .expect("failed to read label");
        if ret != 0 || label.is_null() {
            log::error!("failed to read label");
        }
        let label = CStr::from_ptr(label)
            .to_str()
            .expect("failed to read label")
            .to_string();
        t.labels.push(label);
    }

    for a in 0..A {
        if amap[a] >= 0 {
            let mut attr: *mut libc::c_char = std::ptr::null_mut();
            let ret = (*attrs)
                .to_string
                .map(|f| {
                    f(
                        attrs as *mut _,
                        a as i32,
                        &mut attr as *mut *mut _ as *mut *const _,
                    )
                })
                .expect("failed to read label");
            if ret != 0 || attr.is_null() {
                log::error!("failed to read label");
            }
            let attr = CStr::from_ptr(attr)
                .to_str()
                .expect("failed to read label")
                .to_string();
            t.attrs.push(attr);
        }
    }

    for l in 0..L {
        let r = label_refs
            .offset(l as isize)
            .as_ref()
            .expect("failed to read label ref");
        let r: FeatRefs = (0..r.num_features)
            .map(|x| fmap[*r.fids.offset(x as isize) as usize])
            .filter(|fid| *fid >= 0)
            .map(|fid| fid as usize)
            .collect();
        t.label_refs.push(r);
    }

    t.attr_refs.resize(B as usize, FeatRefs::default());
    for a in 0..A {
        if amap[a] >= 0 {
            let r = attr_refs
                .offset(a as isize)
                .as_ref()
                .expect("failed to read attr ref");
            let id = amap[a];
            let r: FeatRefs = (0..r.num_features)
                .map(|x| fmap[*r.fids.offset(x as isize) as usize])
                .filter(|fid| *fid >= 0)
                .map(|fid| fid as usize)
                .collect();
            t.attr_refs[id as usize] = r;
        }
    }
    let path = CStr::from_ptr(fpath)
        .to_str()
        .expect("failed to read filename")
        .to_string();
    let w = File::create(path + ".json").expect("failed to create file");
    serde_json::to_writer_pretty(w, &t).expect("failed to write");
}

impl From<T> for Crf1dModel {
    fn from(value: T) -> Self {
        Self {
            attr_refs: value.attr_refs,
            label_refs: value.label_refs,
            features: value.features,
            labels: Quark::from(value.labels),
            attrs: Quark::from(value.attrs),
        }
    }
}

#[derive(Debug)]
pub struct Crf1dModel {
    attr_refs: Vec<FeatRefs>,
    label_refs: Vec<FeatRefs>,
    features: Vec<Feature>,
    labels: Quark,
    attrs: Quark,
}

impl Crf1dModel {
    pub fn from_json(path: PathBuf) -> Self {
        let f = File::open(path).expect("failed to open file");
        let t: T = serde_json::from_reader(f).expect("failed to read model");
        Self::from(t)
    }

    pub fn from_path(path: PathBuf) -> Self {
        Self::from_memory(std::fs::read(path).expect("failed to read model"))
    }

    pub fn from_memory(buffer: Vec<u8>) -> Self {
        let magic = b"lCRF";
        // assert!(magic.bytes() == buffer.first_chunk::<4>().unwrap());
        let v: Vec<usize> = buffer[16..(16 + 4 * 8)]
            .chunks(4)
            .map(|x| u32::from_le_bytes(x.try_into().unwrap()) as usize)
            .collect();
        let size = u32::from_le_bytes(buffer[4..8].try_into().unwrap()) as usize;
        let n_feats = v[0];
        let n_labels = v[1];
        let n_attrs = v[2];
        let off_feats = v[3];
        let off_labels = v[4];
        let off_attrs = v[5];
        let off_label_refs = v[6];
        let off_attr_refs = v[7];

        const CHUNK_SIZE: usize = 12;
        const FEATURE_SIZE: usize = 20;
        let n_active_feats =
            u32::from_le_bytes(buffer[off_feats + 8..off_feats + 12].try_into().unwrap()) as usize;
        let mut features = Vec::new();
        for i in 0..n_active_feats {
            let offset = off_feats + CHUNK_SIZE + FEATURE_SIZE * i;
            if offset + 20 >= size {
                println!("exceeding size: {size}, {}", offset + 20);
            }
            let cat = u32::from_le_bytes(
                buffer[offset..offset + 4]
                    .try_into()
                    .expect("failed to read cat"),
            );
            let src = u32::from_le_bytes(buffer[offset + 4..offset + 8].try_into().unwrap());
            let dst = u32::from_le_bytes(buffer[offset + 8..offset + 12].try_into().unwrap());
            let weight: f64 =
                f64::from_le_bytes(buffer[offset + 12..offset + 20].try_into().unwrap());
            let f = Feature {
                cat: cat.try_into().expect("invalid feature type"),
                src,
                dst,
                weight,
            };
            features.push(f);
        }
        let mut label_refs = Vec::new();
        let n_active_label_refs = u32::from_le_bytes(
            buffer[off_label_refs + 8..off_label_refs + 12]
                .try_into()
                .unwrap(),
        ) as usize;
        println!("n: {}, n_act: {}", n_labels, n_active_label_refs);
        for i in 0..n_labels {
            let offset = off_label_refs + CHUNK_SIZE + 4 * i;
            let offset =
                u32::from_le_bytes(buffer[offset..offset + 4].try_into().unwrap()) as usize;
            let n = u32::from_le_bytes(buffer[offset..offset + 4].try_into().unwrap()) as usize;
            let v: FeatRefs = (0..n)
                .map(|j| {
                    u32::from_le_bytes(
                        buffer[offset + 4 + 4 * j..offset + 4 + 4 * j + 4]
                            .try_into()
                            .unwrap(),
                    ) as usize
                })
                .collect();
            label_refs.push(v);
        }
        let mut attr_refs = Vec::new();
        let n_active_attr_refs = u32::from_le_bytes(
            buffer[off_attr_refs + 8..off_attr_refs + 12]
                .try_into()
                .unwrap(),
        ) as usize;
        println!("n: {}, n_act: {}", n_attrs, n_attrs);
        for i in 0..n_active_attr_refs {
            let offset = off_attr_refs + CHUNK_SIZE + 4 * i;
            let offset =
                u32::from_le_bytes(buffer[offset..offset + 4].try_into().unwrap()) as usize;
            let n = u32::from_le_bytes(buffer[offset..offset + 4].try_into().unwrap()) as usize;
            let v: FeatRefs = (0..n)
                .map(|j| {
                    u32::from_le_bytes(
                        buffer[offset + 4 + 4 * j..offset + 4 + 4 * j + 4]
                            .try_into()
                            .unwrap(),
                    ) as usize
                })
                .collect();
            attr_refs.push(v);
        }
        let labels: Vec<String> = CQDB::new(&buffer[off_labels as usize..])
            .expect("failed to read labels")
            .iter()
            .map(Result::unwrap)
            .map(|(k, v)| v.to_string())
            .collect();
        let attrs: Vec<String> = CQDB::new(&buffer[off_attrs as usize..])
            .expect("failed to read attrs")
            .iter()
            .map(Result::unwrap)
            .map(|(k, v)| v.to_string())
            .collect();

        Self {
            attr_refs,
            label_refs,
            features: features,
            labels: labels.into(),
            attrs: attrs.into(),
        }
    }

    pub fn num_labels(&self) -> usize {
        self.labels.len()
    }

    pub fn num_attrs(&self) -> usize {
        self.attrs.len()
    }

    pub(crate) fn crf1dm_get_labelref(&self, lid: usize) -> &FeatRefs {
        &self.label_refs[lid]
    }

    pub fn crf1dm_get_attrref(&self, aid: usize) -> &FeatRefs {
        &self.attr_refs[aid]
    }

    pub(crate) fn crf1dm_get_featureid(&self, refx: &FeatRefs, i: usize) -> usize {
        refx[i]
    }

    pub(crate) fn crf1dm_get_feature(&self, fid: usize) -> &Feature {
        &self.features[fid]
    }

    fn dump(&self) {
        println!("TRANSITIONS:");
        for i in 0..self.num_labels() {
            let refs = self.crf1dm_get_labelref(i);
            for j in 0..refs.len() {
                let fid = self.crf1dm_get_featureid(refs, j);
                let f = self.crf1dm_get_feature(fid);
                let from = self.labels.to_str(f.src as usize).unwrap_or("NULL");
                let to = self.labels.to_str(f.dst as usize).unwrap_or("NULL");
                println!("({:?}) {} -> {}: {:.4}", f.cat, from, to, f.weight);
            }
        }
        /* Dump the transition features. */
        println!("STATE_FEATURES:");
        for i in 0..self.num_attrs() {
            let refs = self.crf1dm_get_attrref(i);
            for j in 0..refs.len() {
                let fid = self.crf1dm_get_featureid(refs, j);
                let f = self.crf1dm_get_feature(fid);
                assert!(f.src as usize == i, "WARNING: an inconsistent attribute reference.");
                let from = self.attrs.to_str(f.src as usize).unwrap_or("NULL");
                let to = self.labels.to_str(f.dst as usize).unwrap_or("NULL");
                println!("({:?}) {} -> {}: {:.4}", f.cat, from, to, f.weight);
            }
        }
    }
}

impl Model for Crf1dModel {
    fn get_tagger(&self) -> Crf1dTagger {
        Crf1dTagger::new(self)
    }

    fn get_labels(&self) -> &Quark {
        &self.labels
    }

    fn get_attrs(&self) -> &Quark {
        &self.attrs
    }

    fn load(path: PathBuf) -> Self {
        Self::from_path(path)
    }

    fn save(&self, path: std::path::PathBuf) {
        let w = File::create(path).expect("failed to create file");
        // serde_json::to_writer(w, self).expect("failed to write");
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    #[test]
    fn json() {
        let start = Instant::now();
        let path = "ner.json";
        let f = File::open(path).expect("failed to open file");
        let t: T = serde_json::from_reader(f).expect("failed to parse model");
        let model = Crf1dModel::from(t);
        println!("took {:?} to load model", start.elapsed());
        assert_eq!(model.attr_refs.len(), 22033);
        assert_eq!(model.attrs.len(), 22033);
        assert_eq!(model.label_refs.len(), 9);
        assert_eq!(model.labels.len(), 9);
        assert_eq!(model.features.len(), 29169);
    }

    #[test]
    fn dump() {
        let path = "ner";
        let model = Crf1dModel::from_path(path.into());
        model.dump();
    }
}
