//! Object manager exports: objects of a title's own types, and opening and referencing objects
//! by name or pointer.
//!
//! Copied from the author's original anybox-tk code (its `system.rs`), with the author's
//! permission, and adapted to this runtime's kernel.

use super::nt::{attributes_name, object_pointer};
use super::value;
use crate::kernel::object::Body;
use crate::kernel::{Call, CallResult, KResult, Kernel, status};

/// `ObCreateObject(ObjectType, ObjectAttributes, ObjectBodySize, *Object)`: a zeroed body of the
/// title's own type. Its name, if the attributes give one, is entered by `ObInsertObject`.
pub fn create_object(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (size, out) = (c.arg(2), c.arg(3));
    let body = k.heap.alloc(size.max(4));
    if body == 0 {
        return value(status::NO_MEMORY);
    }
    k.objects.add_at(body, Body::Other);
    k.w32(out, body)?;
    value(status::SUCCESS)
}

/// `ObInsertObject(Object, ObjectAttributes, ObjectPointerBias, *Handle)`: names the object, if
/// the attributes say how, and opens a handle on it.
pub fn insert_object(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (object, attributes, out) = (c.arg(0), c.arg(1), c.arg(3));
    let Some(id) = k.objects.at(object) else {
        return value(status::INVALID_PARAMETER);
    };
    if let Some(name) = attributes_name(k, attributes)? {
        if k.objects.named(&name).is_some() {
            return value(status::OBJECT_NAME_COLLISION);
        }
        k.objects.set_name(id, &name);
    }
    let handle = k.objects.open(id);
    k.w32_opt(out, handle)?;
    value(status::SUCCESS)
}

/// `ObOpenObjectByName(ObjectAttributes, ObjectType, ParseContext, *Handle)`.
pub fn open_object_by_name(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(name) = attributes_name(k, c.arg(0))? else {
        return value(status::INVALID_PARAMETER);
    };
    let Some(id) = k.objects.named(&name) else {
        return value(status::OBJECT_NAME_NOT_FOUND);
    };
    let handle = k.objects.open(id);
    k.w32(c.arg(3), handle)?;
    value(status::SUCCESS)
}

/// `ObOpenObjectByPointer(Object, ObjectType, *Handle)`.
pub fn open_object_by_pointer(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(id) = k.objects.at(c.arg(0)) else {
        return value(status::INVALID_PARAMETER);
    };
    let handle = k.objects.open(id);
    k.w32(c.arg(2), handle)?;
    value(status::SUCCESS)
}

/// `ObReferenceObjectByName(*ObjectName, Attributes, ObjectType, ParseContext, *Object)`: an
/// object by name, devices the title created among them.
pub fn reference_object_by_name(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let name = k.ansi_string(c.arg(0))?;
    let Some(id) = k.objects.named(&name) else {
        return value(status::OBJECT_NAME_NOT_FOUND);
    };
    let Some(address) = object_pointer(k, id) else {
        return value(status::NO_MEMORY);
    };
    k.w32(c.arg(4), address)?;
    value(status::SUCCESS)
}
