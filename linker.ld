ENTRY(_start)

MEMORY {
    RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 128M
}

SECTIONS {
    .text : {
        *(.text.boot)
        *(.text)
        *(.rodata)
    } > RAM
    .data : { *(.data) } > RAM
    .bss : { *(.bss) *(COMMON) } > RAM
    PROVIDE(_stack_top = ORIGIN(RAM) + LENGTH(RAM));
}
