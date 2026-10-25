//! The kernel call handlers, by export name.
//!
//! Every handler has the `Handler` signature, including those that cannot fail.
#![allow(clippy::unnecessary_wraps)]

mod ex;
mod hal;
mod io;
pub(crate) mod ke;
mod mm;
mod nt;
mod rtl;
#[cfg(test)]
mod tests;

// Copied from anybox-tk with the owner's permission.
mod exec;
mod printf;
mod rtl_misc;
mod strings;
mod system;
mod xc;

// Copied from anybox-tk with the owner's permission: IRPs, queues, APCs, Nt objects, Ob.
mod apc;
pub(crate) mod irp;
mod ntobjects;
mod ob;
#[cfg(test)]
mod ported_tests;
mod queue;

pub use ex::default_eeprom;
pub use ke::release_mutant_object;

use super::{CallResult, Handler, KResult};

/// A 32-bit result.
fn value(v: u32) -> KResult<CallResult> {
    Ok(CallResult::Value(v))
}

/// A void call, or one that has nothing to do.
pub(crate) fn nothing(_: &mut super::Kernel, _: &super::Call) -> KResult<CallResult> {
    value(0)
}

/// The handler of an export, by name. One table, grouped by subsystem, so exports that do
/// nothing are listed where they belong rather than in one arm.
#[must_use]
#[allow(clippy::too_many_lines, clippy::match_same_arms)]
pub fn lookup(name: &str) -> Option<Handler> {
    let handler: Handler = match name {
        // Executive: pool, settings, interlocked operations.
        "ExAllocatePool" | "ExAllocatePoolWithTag" => ex::allocate_pool,
        "ExFreePool" => ex::free_pool,
        "ExQueryPoolBlockSize" => ex::query_pool_block_size,
        "ExQueryNonVolatileSetting" => ex::query_non_volatile_setting,
        "ExSaveNonVolatileSetting" => ex::save_non_volatile_setting,
        "InterlockedCompareExchange" => ex::interlocked_compare_exchange,
        "InterlockedIncrement" => ex::interlocked_increment,
        "InterlockedDecrement" => ex::interlocked_decrement,
        "InterlockedExchange" => ex::interlocked_exchange,
        "InterlockedExchangeAdd" => ex::interlocked_exchange_add,
        "InterlockedPushEntrySList" => ex::slist_push,
        "InterlockedPopEntrySList" => ex::slist_pop,
        "InterlockedFlushSList" => ex::slist_flush,
        "ExInterlockedAddLargeInteger" => ex::add_large_integer,
        "ExInterlockedAddLargeStatistic" => ex::add_large_statistic,
        "ExInterlockedCompareExchange64" => ex::compare_exchange_64,
        "ExfInterlockedInsertHeadList" => ex::insert_head_list,
        "ExfInterlockedInsertTailList" => ex::insert_tail_list,
        "ExfInterlockedRemoveHeadList" => ex::remove_head_list,

        // Kernel: events, timers, DPCs, interrupts, waits, IRQL, time, threads.
        "KeInitializeEvent" => ke::initialize_event,
        "KeSetEvent" => ke::set_event,
        "KeResetEvent" => ke::reset_event,
        "KePulseEvent" => ke::pulse_event,
        "KeSetEventBoostPriority" => ke::set_event_boost_priority,
        "KeWaitForSingleObject" => ke::wait_for_single_object,
        "KeWaitForMultipleObjects" => ke::wait_for_multiple_objects,
        "KeDelayExecutionThread" => ke::delay_execution_thread,
        "KeStallExecutionProcessor" => ke::stall_execution_processor,
        "KeInitializeTimerEx" => ke::initialize_timer,
        "KeSetTimer" => ke::set_timer,
        "KeSetTimerEx" => ke::set_timer_ex,
        "KeCancelTimer" => ke::cancel_timer,
        "KeInitializeDpc" => ke::initialize_dpc,
        "KeInsertQueueDpc" => ke::insert_queue_dpc,
        "KeRemoveQueueDpc" => ke::remove_queue_dpc,
        "KeIsExecutingDpc" => ke::is_executing_dpc,
        "KeInitializeInterrupt" => ke::initialize_interrupt,
        "KeConnectInterrupt" => ke::connect_interrupt,
        "KeDisconnectInterrupt" => ke::disconnect_interrupt,
        "KeInitializeMutant" => ke::initialize_mutant,
        "KeReleaseMutant" => ke::release_mutant,
        "KeInitializeSemaphore" => ke::initialize_semaphore,
        "KeReleaseSemaphore" => ke::release_semaphore,
        "KfRaiseIrql" => ke::raise_irql,
        "KfLowerIrql" | "KiUnlockDispatcherDatabase" => ke::lower_irql,
        "KeRaiseIrqlToDpcLevel" => ke::raise_irql_to_dpc_level,
        "KeRaiseIrqlToSynchLevel" => ke::raise_irql_to_synch_level,
        "KeGetCurrentIrql" => ke::get_current_irql,
        "KeQuerySystemTime" => ke::query_system_time,
        "KeQueryInterruptTime" => ke::query_interrupt_time,
        "KeQueryPerformanceCounter" => ke::query_performance_counter,
        "KeQueryPerformanceFrequency" => ke::query_performance_frequency,
        "KeGetCurrentThread" => ke::get_current_thread,
        "KeSetBasePriorityThread" => ke::set_base_priority_thread,
        "KeQueryBasePriorityThread" => ke::query_base_priority_thread,
        "KeSetPriorityThread" => ke::set_priority_thread,
        "KeSetDisableBoostThread" | "KeSetPriorityProcess" | "KeBoostPriorityThread" => nothing,
        "KeEnterCriticalRegion" | "KeLeaveCriticalRegion" => nothing,
        "KeAlertThread" | "KeTestAlertThread" => nothing,
        "KeSaveFloatingPointState" | "KeRestoreFloatingPointState" => nothing,
        "KeBugCheck" | "KeBugCheckEx" => ke::bug_check,
        "KeSynchronizeExecution" => ke::synchronize_execution,
        "KeSuspendThread" => ke::suspend_thread,
        "KeResumeThread" | "KeAlertResumeThread" => ke::resume_thread,

        // Processes and threads.
        "PsCreateSystemThreadEx" => ke::create_system_thread_ex,
        "PsCreateSystemThread" => ke::create_system_thread,
        "PsTerminateSystemThread" => ke::terminate_system_thread,
        "PsSetCreateThreadNotifyRoutine" => nothing,
        "PsQueryStatistics" => ke::query_statistics,

        // Memory manager.
        "MmAllocateContiguousMemory" | "MmAllocateSystemMemory" => mm::allocate_contiguous,
        "MmAllocateContiguousMemoryEx" => mm::allocate_contiguous_ex,
        "MmFreeContiguousMemory" => mm::free_contiguous,
        "MmFreeSystemMemory" => mm::free_system,
        "MmClaimGpuInstanceMemory" => mm::claim_gpu_instance_memory,
        "MmPersistContiguousMemory"
        | "MmSetAddressProtect"
        | "MmLockUnlockBufferPages"
        | "MmLockUnlockPhysicalPage" => nothing,
        "MmGetPhysicalAddress" => mm::get_physical_address,
        "MmQueryAllocationSize" => mm::query_allocation_size,
        "MmQueryAddressProtect" => mm::query_address_protect,
        "MmQueryStatistics" => mm::query_statistics,
        "MmIsAddressValid" => mm::is_address_valid,
        "MmMapIoSpace" => mm::map_io_space,
        "MmUnmapIoSpace" | "MmDbgReleaseAddress" => nothing,
        "MmCreateKernelStack" => mm::create_kernel_stack,
        "MmDeleteKernelStack" => mm::delete_kernel_stack,
        "MmDbgAllocateMemory" | "MmDbgFreeMemory" | "MmDbgQueryAvailablePages" => nothing,
        "NtAllocateVirtualMemory" => mm::allocate_virtual,
        "NtFreeVirtualMemory" => mm::free_virtual,
        "NtQueryVirtualMemory" => mm::query_virtual,
        "NtProtectVirtualMemory" => mm::protect_virtual,
        "XeLoadSection" => mm::load_section,
        "XeUnloadSection" => mm::unload_section,

        // NT object services.
        "NtClose" => nt::close,
        "NtCreateEvent" => nt::create_event,
        "NtSetEvent" => nt::set_event,
        "NtClearEvent" => nt::clear_event,
        "NtPulseEvent" => nt::pulse_event,
        "NtQueryEvent" => nt::query_event,
        "NtWaitForSingleObject" => nt::wait_for_single_object,
        "NtWaitForSingleObjectEx" => nt::wait_for_single_object_ex,
        "NtWaitForMultipleObjectsEx" => nt::wait_for_multiple_objects_ex,
        "NtCreateSemaphore" => nt::create_semaphore,
        "NtReleaseSemaphore" => nt::release_semaphore,
        "NtCreateMutant" => nt::create_mutant,
        "NtReleaseMutant" => nt::release_mutant,
        "NtDuplicateObject" => nt::duplicate_object,
        "NtYieldExecution" => nt::yield_execution,
        "NtSuspendThread" => nt::suspend_thread,
        "NtResumeThread" => nt::resume_thread,
        "NtSetSystemTime" => nt::set_system_time,
        "ObReferenceObjectByHandle" => nt::reference_object_by_handle,
        "ObfDereferenceObject" => nt::dereference_object,
        "ObfReferenceObject" | "ObMakeTemporaryObject" => nothing,
        "ObReferenceObjectByPointer" => nothing,

        // I/O and the file system.
        "NtCreateFile" => io::create_file,
        "IoCreateFile" => io::io_create_file,
        "NtOpenFile" => io::open,
        "NtReadFile" => io::read_file,
        "NtWriteFile" => io::write_file,
        "NtQueryInformationFile" => io::query_information_file,
        "NtSetInformationFile" => io::set_information_file,
        "NtQueryVolumeInformationFile" => io::query_volume_information_file,
        "NtQueryDirectoryFile" => io::query_directory_file,
        "NtQueryFullAttributesFile" => io::query_full_attributes_file,
        "NtDeleteFile" => io::delete_file,
        "NtFlushBuffersFile" => io::flush_buffers_file,
        "NtDeviceIoControlFile" | "NtFsControlFile" => io::device_io_control_file,
        "IoCreateSymbolicLink" => io::create_symbolic_link,
        "IoDeleteSymbolicLink" => io::delete_symbolic_link,
        "IoCreateDevice" => io::create_device,
        "IoDismountVolume" | "IoDismountVolumeByName" => nothing,

        // Run-time library.
        "RtlInitAnsiString" => rtl::init_ansi_string,
        "RtlInitUnicodeString" => rtl::init_unicode_string,
        "RtlEqualString" => rtl::equal_string,
        "RtlCompareString" => rtl::compare_string,
        "RtlCopyString" => rtl::copy_string,
        "RtlUpperString" => rtl::upper_string,
        "RtlAppendStringToString" => rtl::append_string_to_string,
        "RtlFreeAnsiString" | "RtlFreeUnicodeString" => rtl::free_string,
        "RtlAnsiStringToUnicodeString" => rtl::ansi_string_to_unicode_string,
        "RtlUnicodeStringToAnsiString" => rtl::unicode_string_to_ansi_string,
        "RtlNtStatusToDosError" => rtl::nt_status_to_dos_error,
        "RtlInitializeCriticalSection" => rtl::initialize_critical_section,
        "RtlEnterCriticalSection" | "RtlEnterCriticalSectionAndRegion" => {
            rtl::enter_critical_section
        }
        "RtlLeaveCriticalSection" | "RtlLeaveCriticalSectionAndRegion" => {
            rtl::leave_critical_section
        }
        "RtlTryEnterCriticalSection" => rtl::try_enter_critical_section,
        "RtlTimeToTimeFields" => rtl::time_to_time_fields,
        "RtlTimeFieldsToTime" => rtl::time_fields_to_time,
        "RtlCompareMemory" => rtl::compare_memory,
        "RtlCompareMemoryUlong" => rtl::compare_memory_ulong,
        "RtlFillMemory" => rtl::fill_memory,
        "RtlFillMemoryUlong" => rtl::fill_memory_ulong,
        "RtlZeroMemory" => rtl::zero_memory,
        "RtlMoveMemory" => rtl::move_memory,
        "RtlUlongByteSwap" => rtl::ulong_byte_swap,
        "RtlUshortByteSwap" => rtl::ushort_byte_swap,
        "RtlExtendedIntegerMultiply" => rtl::extended_integer_multiply,
        "RtlExtendedLargeIntegerDivide" => rtl::extended_large_integer_divide,
        "RtlUpperChar" => rtl::upper_char,
        "RtlLowerChar" => rtl::lower_char,
        "RtlCaptureStackBackTrace" | "RtlWalkFrameChain" => nothing,
        "RtlGetCallersAddress" => rtl::get_callers_address,
        "RtlAssert" => rtl::assert,
        "RtlRip" | "RtlRaiseException" | "RtlRaiseStatus" | "RtlUnwind" | "ExRaiseException"
        | "ExRaiseStatus" => rtl::raise,

        // HAL, AV, debugging.
        "HalGetInterruptVector" => hal::get_interrupt_vector,
        "HalReadWritePCISpace" => hal::read_write_pci_space,
        "HalRegisterShutdownNotification"
        | "HalEnableSystemInterrupt"
        | "HalDisableSystemInterrupt" => nothing,
        "HalReadSMBusValue" => hal::read_smbus_value,
        "HalWriteSMBusValue" | "HalWriteSMCScratchRegister" => nothing,
        "HalReadSMCTrayState" => hal::read_smc_tray_state,
        "HalIsResetOrShutdownPending" => nothing,
        "HalInitiateShutdown" | "HalReturnToFirmware" => hal::halt,
        "HalClearSoftwareInterrupt"
        | "HalRequestSoftwareInterrupt"
        | "HalEnableSecureTrayEject" => nothing,
        "PhyGetLinkState" | "PhyInitialize" => nothing,
        "AvGetSavedDataAddress" => hal::av_get_saved_data_address,
        "AvSetSavedDataAddress" => hal::av_set_saved_data_address,
        "AvSendTVEncoderOption" => hal::av_send_tv_encoder_option,
        "AvSetDisplayMode" => hal::av_set_display_mode,
        "DbgPrint" => hal::dbg_print,
        "DbgBreakPoint"
        | "DbgBreakPointWithStatus"
        | "DbgLoadImageSymbols"
        | "DbgUnLoadImageSymbols"
        | "DbgPrompt" => nothing,
        // Copied from anybox-tk with the owner's permission: crypto, printf, strings, Rtl
        // arithmetic and contexts, read/write locks, port buffers, the file system cache.
        "XcSHAInit" => xc::sha_init,
        "XcSHAUpdate" => xc::sha_update,
        "XcSHAFinal" => xc::sha_final,
        "XcRC4Key" => xc::rc4_key,
        "XcRC4Crypt" => xc::rc4_crypt,
        "XcHMAC" => xc::hmac,
        "XcPKEncPublic" => xc::pk_enc_public,
        "XcPKDecPrivate" => xc::pk_dec_private,
        "XcPKGetKeyLen" => xc::pk_get_key_len,
        "XcVerifyPKCS1Signature" => xc::verify_pkcs1_signature,
        "XcModExp" => xc::mod_exp,
        "XcDESKeyParity" => xc::des_key_parity,
        "XcKeyTable" => xc::key_table,
        "XcBlockCrypt" => xc::block_crypt,
        "XcBlockCryptCBC" => xc::block_crypt_cbc,
        "XcCryptService" => xc::crypt_service,
        "XcUpdateCrypto" => xc::update_crypto,
        "RtlSnprintf" => printf::snprintf,
        "RtlSprintf" => printf::sprintf,
        "RtlVsnprintf" => printf::vsnprintf,
        "RtlVsprintf" => printf::vsprintf,
        "RtlAppendUnicodeStringToString" => strings::append_unicode_string_to_string,
        "RtlAppendUnicodeToString" => strings::append_unicode_to_string,
        "RtlCharToInteger" => strings::char_to_integer,
        "RtlCompareUnicodeString" => strings::compare_unicode_string,
        "RtlCopyUnicodeString" => strings::copy_unicode_string,
        "RtlCreateUnicodeString" => strings::create_unicode_string,
        "RtlDowncaseUnicodeChar" => strings::downcase_unicode_char,
        "RtlDowncaseUnicodeString" => strings::downcase_unicode_string,
        "RtlEqualUnicodeString" => strings::equal_unicode_string,
        "RtlIntegerToChar" => strings::integer_to_char,
        "RtlIntegerToUnicodeString" => strings::integer_to_unicode_string,
        "RtlMultiByteToUnicodeN" => strings::multi_byte_to_unicode_n,
        "RtlMultiByteToUnicodeSize" => strings::multi_byte_to_unicode_size,
        "RtlUnicodeStringToInteger" => strings::unicode_string_to_integer,
        "RtlUnicodeToMultiByteN" => strings::unicode_to_multi_byte_n,
        "RtlUnicodeToMultiByteSize" => strings::unicode_to_multi_byte_size,
        "RtlUpcaseUnicodeChar" => strings::upcase_unicode_char,
        "RtlUpcaseUnicodeString" => strings::upcase_unicode_string,
        "RtlUpcaseUnicodeToMultiByteN" => strings::upcase_unicode_to_multi_byte_n,
        "RtlCaptureContext" => rtl_misc::capture_context,
        "RtlExtendedMagicDivide" => rtl_misc::extended_magic_divide,
        "RtlMapGenericMask" => rtl_misc::map_generic_mask,
        "ExInitializeReadWriteLock" => exec::initialize_read_write_lock,
        "ExAcquireReadWriteLockExclusive" => exec::acquire_read_write_lock_exclusive,
        "ExAcquireReadWriteLockShared" => exec::acquire_read_write_lock_shared,
        "ExReleaseReadWriteLock" => exec::release_read_write_lock,
        "ExReadWriteRefurbInfo" => exec::read_write_refurb_info,
        "READ_PORT_BUFFER_UCHAR" => system::read_port_buffer_uchar,
        "READ_PORT_BUFFER_USHORT" => system::read_port_buffer_ushort,
        "READ_PORT_BUFFER_ULONG" => system::read_port_buffer_ulong,
        "WRITE_PORT_BUFFER_UCHAR" | "WRITE_PORT_BUFFER_USHORT" | "WRITE_PORT_BUFFER_ULONG" => {
            nothing
        }
        "FscGetCacheSize" => system::get_cache_size,
        "FscInvalidateIdleBlocks" => nothing,
        "FscSetCacheSize" => system::set_cache_size,
        // Copied from anybox-tk with the owner's permission: IRPs, queues, APCs, Nt objects, Ob.
        "IoAllocateIrp" => irp::allocate_irp,
        "IoInitializeIrp" => irp::initialize_irp,
        "IoFreeIrp" => irp::free_irp,
        "IoBuildAsynchronousFsdRequest" => irp::build_asynchronous_fsd_request,
        "IoBuildSynchronousFsdRequest" => irp::build_synchronous_fsd_request,
        "IoBuildDeviceIoControlRequest" => irp::build_device_io_control_request,
        "IofCallDriver" => irp::call_driver_export,
        "IofCompleteRequest" => irp::complete_request_export,
        "IoInvalidDeviceRequest" => irp::invalid_device_request,
        "IoMarkIrpMustComplete" => irp::mark_irp_must_complete,
        "IoQueueThreadIrp" => nothing,
        "IoSynchronousDeviceIoControlRequest" => irp::synchronous_device_io_control_request,
        "IoSynchronousFsdRequest" => irp::synchronous_fsd_request,
        "IoStartPacket" => irp::start_packet,
        "IoStartNextPacket" => irp::start_next_packet,
        "IoStartNextPacketByKey" => irp::start_next_packet_by_key,
        "IoCheckShareAccess" => irp::check_share_access,
        "IoSetShareAccess" => irp::set_share_access,
        "IoRemoveShareAccess" => irp::remove_share_access,
        "IoQueryFileInformation" => irp::query_file_information,
        "IoQueryVolumeInformation" => irp::query_volume_information,
        "IoSetIoCompletion" => irp::set_io_completion,
        "IoDeleteDevice" => irp::delete_device,
        "KeInitializeQueue" => queue::initialize_queue,
        "KeInsertQueue" => queue::insert_queue_tail,
        "KeInsertHeadQueue" => queue::insert_head_queue,
        "KeRemoveQueue" => queue::remove_queue,
        "KeRundownQueue" => queue::rundown_queue,
        "KeInitializeDeviceQueue" => queue::initialize_device_queue,
        "KeInsertDeviceQueue" => queue::insert_device_queue_tail,
        "KeInsertByKeyDeviceQueue" => queue::insert_by_key_device_queue,
        "KeRemoveDeviceQueue" => queue::remove_device_queue,
        "KeRemoveByKeyDeviceQueue" => queue::remove_by_key_device_queue,
        "KeRemoveEntryDeviceQueue" => queue::remove_entry_device_queue,
        "KeInitializeApc" => apc::initialize_apc,
        "KeInsertQueueApc" => apc::insert_queue_apc,
        "NtQueueApcThread" => apc::queue_apc_thread,
        "NtUserIoApcDispatcher" => apc::user_io_apc_dispatcher,
        "NtQueryMutant" => ntobjects::query_mutant,
        "NtQuerySemaphore" => ntobjects::query_semaphore,
        "NtCreateTimer" => ntobjects::create_timer,
        "NtSetTimerEx" => ntobjects::set_timer_ex,
        "NtCancelTimer" => ntobjects::cancel_timer,
        "NtQueryTimer" => ntobjects::query_timer,
        "NtCreateIoCompletion" => ntobjects::create_io_completion,
        "NtSetIoCompletion" => ntobjects::set_io_completion,
        "NtRemoveIoCompletion" => ntobjects::remove_io_completion,
        "NtQueryIoCompletion" => ntobjects::query_io_completion,
        "NtCreateDirectoryObject" => ntobjects::create_directory_object,
        "NtOpenDirectoryObject" => ntobjects::open_directory_object,
        "NtQueryDirectoryObject" => ntobjects::query_directory_object,
        "NtOpenSymbolicLinkObject" => ntobjects::open_symbolic_link_object,
        "NtQuerySymbolicLinkObject" => ntobjects::query_symbolic_link_object,
        "NtSignalAndWaitForSingleObjectEx" => ntobjects::signal_and_wait_for_single_object_ex,
        "NtReadFileScatter" => io::read_file_scatter,
        "NtWriteFileGather" => io::write_file_gather,
        "ObCreateObject" => ob::create_object,
        "ObInsertObject" => ob::insert_object,
        "ObOpenObjectByName" => ob::open_object_by_name,
        "ObOpenObjectByPointer" => ob::open_object_by_pointer,
        "ObReferenceObjectByName" => ob::reference_object_by_name,
        "MmDbgWriteCheck" => mm::dbg_write_check,
        _ => return None,
    };
    Some(handler)
}

/// Whether an export has never been exercised by a title, so its first call is reported (spec
/// 8.1). Only the exports the test titles are known to use correctly count as tested.
#[must_use]
pub fn is_untested(name: &str) -> bool {
    !matches!(
        name,
        "AvGetSavedDataAddress"
            | "AvSendTVEncoderOption"
            | "AvSetDisplayMode"
            | "AvSetSavedDataAddress"
            | "ExAllocatePool"
            | "ExFreePool"
            | "ExQueryNonVolatileSetting"
            | "FscGetCacheSize"
            | "FscSetCacheSize"
            | "HalGetInterruptVector"
            | "HalReadWritePCISpace"
            | "HalRegisterShutdownNotification"
            | "IoCreateDevice"
            | "IoCreateSymbolicLink"
            | "KeConnectInterrupt"
            | "KeInitializeDpc"
            | "KeInitializeInterrupt"
            | "KeInitializeTimerEx"
            | "KeInsertQueueDpc"
            | "KeQueryBasePriorityThread"
            | "KeQuerySystemTime"
            | "KeSetBasePriorityThread"
            | "KeSetDisableBoostThread"
            | "KeSetEvent"
            | "KeSetTimer"
            | "KeWaitForSingleObject"
            | "MmAllocateContiguousMemory"
            | "MmAllocateContiguousMemoryEx"
            | "MmAllocateSystemMemory"
            | "MmClaimGpuInstanceMemory"
            | "MmFreeContiguousMemory"
            | "MmFreeSystemMemory"
            | "MmGetPhysicalAddress"
            | "MmLockUnlockBufferPages"
            | "MmPersistContiguousMemory"
            | "MmQueryAllocationSize"
            | "MmQueryStatistics"
            | "MmSetAddressProtect"
            | "NtAllocateVirtualMemory"
            | "NtClose"
            | "NtCreateEvent"
            | "NtCreateFile"
            | "NtDeviceIoControlFile"
            | "NtFreeVirtualMemory"
            | "NtFsControlFile"
            | "NtOpenFile"
            | "NtQueryDirectoryFile"
            | "NtQueryFullAttributesFile"
            | "NtQueryInformationFile"
            | "NtQueryVolumeInformationFile"
            | "NtReadFile"
            | "NtResumeThread"
            | "NtSetEvent"
            | "NtSetInformationFile"
            | "NtSuspendThread"
            | "NtWaitForSingleObjectEx"
            | "NtWriteFile"
            | "ObReferenceObjectByHandle"
            | "ObfDereferenceObject"
            | "PsCreateSystemThreadEx"
            | "PsTerminateSystemThread"
            | "RtlEnterCriticalSection"
            | "RtlEqualString"
            | "RtlInitAnsiString"
            | "RtlInitializeCriticalSection"
            | "RtlLeaveCriticalSection"
            | "RtlNtStatusToDosError"
            | "RtlTimeToTimeFields"
            | "XcSHAFinal"
            | "XcSHAInit"
            | "XcSHAUpdate"
            | "XeLoadSection"
            | "XeUnloadSection"
    )
}
