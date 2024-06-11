use std::{collections::HashMap, convert::TryInto, path::PathBuf};

use cqdb::CQDB;

use crate::{crf::model::Model, quark::Quark};

pub struct FeatRefs {
    pub fids: Vec<usize>,
    pub num_features: usize,
}
impl FeatRefs {
    pub fn new(n: usize, v: Vec<usize>) -> Self {
        Self {
            fids: v,
            num_features: n,
        }
    }
}

#[derive(Debug)]
pub enum FeatCat {}

#[derive(Debug)]
pub struct Feature {
    pub cat: usize,
    pub src: usize,
    pub dst: usize,
    pub weight: f64,
}

pub struct Crf1dModel {
    buffer: Vec<u8>,
    attr_refs: Vec<FeatRefs>,
    label_refs: Vec<FeatRefs>,
    features: Vec<Feature>,
    labels: Quark,
    attrs: Quark,
}

impl Crf1dModel {
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
        println!("sz: {size}, n_feats: {:?}, num_labels: {:?}, n_attrs: {:?} off_feats: {off_feats}, o_l: {off_labels}, o_a: {off_attrs}, o_l_r: {off_label_refs}, o_a_r: {off_attr_refs}", n_feats, n_labels, n_attrs);

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
                cat: cat as usize,
                src: src as usize,
                dst: dst as usize,
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
            let n = u32::from_le_bytes(buffer[offset + 4..offset + 8].try_into().unwrap()) as usize;
            let v = (0..n)
                .map(|j| {
                    u32::from_le_bytes(
                        buffer[offset + 8 + 4 * j..offset + 8 + 4 * j + 4]
                            .try_into()
                            .unwrap(),
                    ) as usize
                })
                .collect::<Vec<_>>();
            label_refs.push(FeatRefs::new(n, v));
        }
        let mut attr_refs = Vec::new();
        let n_active_attr_refs = u32::from_le_bytes(
            buffer[off_attr_refs + 8..off_attr_refs + 12]
                .try_into()
                .unwrap(),
        ) as usize;
        println!("n: {}, n_act: {}", n_attrs, n_active_attr_refs);
        for i in 0..n_attrs {
            let offset = off_attr_refs + CHUNK_SIZE + 4 * i;
            let offset =
                u32::from_le_bytes(buffer[offset..offset + 4].try_into().unwrap()) as usize;
            let n = u32::from_le_bytes(buffer[offset + 4..offset + 8].try_into().unwrap()) as usize;
            let v = (0..n)
                .map(|j| {
                    u32::from_le_bytes(
                        buffer[offset + 8 + 4 * j..offset + 8 + 4 * j + 4]
                            .try_into()
                            .unwrap(),
                    ) as usize
                })
                .collect::<Vec<_>>();
            attr_refs.push(FeatRefs::new(n, v));
        }
        let labels: Vec<(u32, String)> = CQDB::new(&buffer[off_labels as usize..])
            .expect("failed to read labels")
            .iter()
            .map(Result::unwrap)
            .map(|(k, v)| (k, v.to_string()))
            .collect();
        let attrs: Vec<(u32, String)> = CQDB::new(&buffer[off_attrs as usize..])
            .expect("failed to read attrs")
            .iter()
            .map(Result::unwrap)
            .map(|(k, v)| (k, v.to_string()))
            .collect();

        Self {
            buffer,
            attr_refs,
            label_refs,
            features: features,
            labels: Quark::default(),
            attrs: Quark::default(),
        }
    }

    pub fn num_labels(&self) -> usize {
        todo!()
    }

    pub fn num_attrs(&self) -> usize {
        todo!()
    }

    pub(crate) fn crf1dm_get_labelref(&self, lid: usize) -> &FeatRefs {
        &self.label_refs[lid]
    }

    pub fn crf1dm_get_attrref(&self, aid: usize) -> &FeatRefs {
        &self.attr_refs[aid]
    }

    pub(crate) fn crf1dm_get_featureid(&self, refx: &FeatRefs, i: usize) -> usize {
        refx.fids[i]
    }

    pub(crate) fn crf1dm_get_feature(&self, fid: usize) -> &Feature {
        &self.features[fid]
    }
}

impl Model for Crf1dModel {
    fn get_tagger(&self) -> crate::Tagger {
        todo!()
    }

    fn get_labels(&self) -> crate::quark::Quark {
        todo!()
    }

    fn get_attrs(&self) -> crate::quark::Quark {
        todo!()
    }

    fn dump(&self, path: std::path::PathBuf) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_buf() {
        let path = "ner";
        let model = Crf1dModel::from_path(path.into());
    }
}
