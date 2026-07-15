//! GGUF file format parser
//! Reads GGUF files created by convert_piper_to_gguf.py

use anyhow::{Result, Context};
use std::collections::HashMap;
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::Cursor;

/// GGUF file magic number
const GGUF_MAGIC: u32 = 0x46554747; // "GGUF"

/// Supported GGUF version
const GGUF_VERSION: u32 = 3;

/// GGML data types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
}

impl GGMLType {
    fn from_u32(val: u32) -> Result<Self> {
        match val {
            0 => Ok(GGMLType::F32),
            1 => Ok(GGMLType::F16),
            2 => Ok(GGMLType::Q4_0),
            3 => Ok(GGMLType::Q4_1),
            6 => Ok(GGMLType::Q5_0),
            7 => Ok(GGMLType::Q5_1),
            8 => Ok(GGMLType::Q8_0),
            9 => Ok(GGMLType::Q8_1),
            _ => Err(anyhow::anyhow!("Unknown GGML type: {}", val)),
        }
    }
    
    /// Get bytes per element for this type
    pub fn bytes_per_element(&self) -> usize {
        match self {
            GGMLType::F32 => 4,
            GGMLType::F16 => 2,
            GGMLType::Q4_0 => 18, // 32 values -> 16 bytes + 2 bytes scale
            GGMLType::Q4_1 => 18,
            GGMLType::Q5_0 => 22,
            GGMLType::Q5_1 => 22,
            GGMLType::Q8_0 => 34, // 32 values -> 32 bytes + 2 bytes scale
            GGMLType::Q8_1 => 34,
        }
    }
}

/// Metadata value types
#[derive(Debug, Clone)]
pub enum MetadataValue {
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
}

impl MetadataValue {
    pub fn to_string(&self) -> String {
        match self {
            MetadataValue::Int8(v) => v.to_string(),
            MetadataValue::Int16(v) => v.to_string(),
            MetadataValue::Int32(v) => v.to_string(),
            MetadataValue::Int64(v) => v.to_string(),
            MetadataValue::Float32(v) => v.to_string(),
            MetadataValue::Float64(v) => v.to_string(),
            MetadataValue::Bool(v) => v.to_string(),
            MetadataValue::String(v) => v.clone(),
            MetadataValue::Array(v) => format!("{:?}", v),
        }
    }
}

/// Tensor information
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: GGMLType,
    pub offset: usize,
    pub size: usize,
}

impl TensorInfo {
    /// Calculate number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// GGUF file representation
pub struct GGUFFile {
    pub metadata: HashMap<String, MetadataValue>,
    pub tensors: HashMap<String, TensorInfo>,
}

impl GGUFFile {
    /// Parse GGUF file from memory-mapped data
    pub fn parse(data: &[u8]) -> Result<Self> {
        let mut cursor = Cursor::new(data);
        
        // Read and verify magic
        let magic = cursor.read_u32::<LittleEndian>()
            .context("Failed to read magic number")?;
        if magic != GGUF_MAGIC {
            return Err(anyhow::anyhow!("Invalid GGUF magic: 0x{:08x}", magic));
        }
        
        // Read and verify version
        let version = cursor.read_u32::<LittleEndian>()
            .context("Failed to read version")?;
        if version != GGUF_VERSION {
            return Err(anyhow::anyhow!("Unsupported GGUF version: {}", version));
        }
        
        // Read counts
        let tensor_count = cursor.read_u64::<LittleEndian>()
            .context("Failed to read tensor count")? as usize;
        let metadata_count = cursor.read_u64::<LittleEndian>()
            .context("Failed to read metadata count")? as usize;
        
        tracing::debug!("GGUF: {} tensors, {} metadata entries", tensor_count, metadata_count);
        
        // Read metadata
        let mut metadata = HashMap::new();
        for _ in 0..metadata_count {
            let key = read_string(&mut cursor)?;
            let value = read_metadata_value(&mut cursor)?;
            metadata.insert(key, value);
        }
        
        // Read tensor info
        let mut tensors = HashMap::new();
        for _ in 0..tensor_count {
            let name = read_string(&mut cursor)?;
            let n_dims = cursor.read_u32::<LittleEndian>()? as usize;
            
            // Read shape (in reverse order)
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(cursor.read_u64::<LittleEndian>()? as usize);
            }
            shape.reverse(); // GGML stores in reverse order
            
            // Read type
            let dtype = GGMLType::from_u32(cursor.read_u32::<LittleEndian>()?)?;
            
            // Read offset
            let offset = cursor.read_u64::<LittleEndian>()? as usize;
            
            // Calculate size
            let numel: usize = shape.iter().product();
            let size = match dtype {
                GGMLType::F32 => numel * 4,
                GGMLType::F16 => numel * 2,
                GGMLType::Q8_0 => {
                    // Q8_0: blocks of 32 elements
                    let blocks = (numel + 31) / 32;
                    blocks * 34 // 32 bytes + 2 byte scale per block
                }
                GGMLType::Q4_0 => {
                    let blocks = (numel + 31) / 32;
                    blocks * 18 // 16 bytes + 2 byte scale per block
                }
                _ => return Err(anyhow::anyhow!("Unsupported tensor type: {:?}", dtype)),
            };
            
            let tensor_info = TensorInfo {
                name: name.clone(),
                shape,
                dtype,
                offset,
                size,
            };
            
            tensors.insert(name, tensor_info);
        }
        
        // Calculate data offset (aligned to 32 bytes)
        let header_size = cursor.position() as usize;
        let data_offset = (header_size + 31) & !31; // Align to 32 bytes
        
        // Adjust tensor offsets
        for tensor in tensors.values_mut() {
            tensor.offset += data_offset;
        }
        
        Ok(GGUFFile { metadata, tensors })
    }
}

/// Read a string from the cursor
fn read_string(cursor: &mut Cursor<&[u8]>) -> Result<String> {
    let len = cursor.read_u64::<LittleEndian>()? as usize;
    let mut bytes = vec![0u8; len];
    std::io::Read::read_exact(cursor, &mut bytes)?;
    String::from_utf8(bytes).context("Invalid UTF-8 in string")
}

/// Read a metadata value
fn read_metadata_value(cursor: &mut Cursor<&[u8]>) -> Result<MetadataValue> {
    let value_type = cursor.read_u32::<LittleEndian>()?;
    
    match value_type {
        0 => Ok(MetadataValue::Int8(cursor.read_i8()?)),
        1 => Ok(MetadataValue::Int16(cursor.read_i16::<LittleEndian>()?)),
        2 => Ok(MetadataValue::Int32(cursor.read_i32::<LittleEndian>()?)),
        3 => Ok(MetadataValue::Int64(cursor.read_i64::<LittleEndian>()?)),
        4 => Ok(MetadataValue::Int32(cursor.read_i32::<LittleEndian>()?)), // uint32 as int32
        5 => Ok(MetadataValue::Int64(cursor.read_i64::<LittleEndian>()?)), // uint64 as int64
        6 => Ok(MetadataValue::Float32(cursor.read_f32::<LittleEndian>()?)),
        7 => Ok(MetadataValue::Bool(cursor.read_u8()? != 0)),
        8 => Ok(MetadataValue::String(read_string(cursor)?)),
        9 => {
            // Array type
            let array_type = cursor.read_u32::<LittleEndian>()?;
            let array_len = cursor.read_u64::<LittleEndian>()? as usize;
            let mut array = Vec::with_capacity(array_len);
            
            // For now, we'll skip complex array parsing
            // In production, this would recursively parse array elements
            Ok(MetadataValue::Array(array))
        }
        _ => Err(anyhow::anyhow!("Unknown metadata value type: {}", value_type)),
    }
}