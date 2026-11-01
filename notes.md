DATA TYPES USAGE
1. int - used for CPU logic: loop control, array indices and boolean logic.
2. int8_t/int16_t/int32_t: for memory payloads like weights, buffers and structs.
3. unsigned by default for hardware and bitwise logic
4. signed only for physical measurement
5. const: pushes large data to flash as read only (.rodata), otherwise all will be in SRAM. Also use for Read only protection
6. volatile: ensures CPU fetches new data from the phsycial memory address instead of cached old value
7. static: ensures data is in SRAM and not in stack. Ensures function restricted only to the specific .c file.

[Higher Address] Stack, Dynamic data in heap, data segment (.data), BSS Segment (.bss), text segment (.text), Reserved [Low addresses]


headers - defind functions and global data

Include guards with preprocessor directives

```c
#ifndef ML_HEADER_H
#define ML_HEADER_H
// Code here
#endif
```

The above guard ensures that if the header file is included twice in the same file.
Say one other library within it uses this current header, in the main file it might get copied twice.
To prevent this the guard is created to ensure only one time copying.


why 1d flattened array instead of 2D?
C forces to hardcode the inner dimesion size directly into every function definition
For eg. void dense_layer(int8_t weights[][784] ....)
C already treats 2D arrays as 1D with the memory address forumla
Easier to format weights as 1D array from python trained weights


How to quantize?
The way weights are quantized first is 
effective scale = (input scale x multiplier) / 2^shift

how python quantizes?
the weights are given in floating point numbers (x)
x = S x (q - Z) where x is the original float, S is scale and Z is zero point, Z = 0
S = max(|w|)/127 to get the step size, if we do this q only ranges between -127 to 127
this works similar to an exchange rate, the exchange rate of rupee to dollor is max value of one dollor  (1)/ max value of dollar in rupee (100)
This compresses the entire weight float range to 127, making each integer increment an increment of S in float

q in int = round(Wfloat/Sw)
this quantizes the weight to make q only between -127 to 127

The accumulated value in one layer is now 32 bits and must similarly be quantized to 8 bits for next layer.
This is done by using the same scaling factor for that layer from python.
Since the S is a float


Makefile USAGE
Looks at timestamp of edit of files and compiles accordingly only the edited ones
Variables are used usually with Capitals, $(Var) allows to access the variable's value
Rule's syntax:
target: prerequisites
<TAB>command

if prerequisites do not exist, it builds them first
clean is not a target but a utility rule, typing make clean applies the command

-Wall: warning all; searches for uninitilized variables, unused variables and type mismathces
-O2: optimization level 2; does register caching, dead code elimination, loop unrolling
-c: compile only, only creates object files and not the final executable; if not used when compiling
ml_engine.c, since main function is not there it will throw an error but the file itself is not executable
-o: output, renames output files


compiler - takes c code and translated to object files
linked, ld: takes object files, calculates memory addresses and stitches them into .elf files
the .ld file is a linker script, dictating physical memory map based on hardware
GNU linker
Two primary commands - MEMORY and SECTIONS
formatters: * is a wildcard, > is routing, telling inker to move data into some space

## LINKER

MEMORY {
    RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 128M
}

Memory section, define a custom name to block called RAM with read, write and eXecute permission
the start os 0x80000000 as per QEMU start of virtual RAM, 128 MBytes of RAM

SECTIONS {
    .text : {
        *(.text.boot)
        *(.text)
        *(.rodata)
    } > RAM
    .data : {*(.data)} > RAM
    .bss : {*(.bss)*(COMMON)} > RAM
    PROVIDE(_stack_top = ORIGIN(RAM) + LENGTH(RAM));
}

.text, define a block named text with the wake up sequence
NOTE there must be space before and after the : next to .text
*(.text.boot) implies to look for every .o object file and extract code tagged
with .text.boot name i.e. boot.S which gives the boot sequence
the *(.text) extracts all text such as main.o and ml_engine.o and stacks underneath bootloader.
the last is *(rodata) which includes all read only data such as const variables, ML arrays and literals

gcc and other compilers auto tag main.c and ml_engine.c and any const variable is taggged as .rodata
static tagged with .data
uninitilized static with .bss
All are pushed onto RAM because it is used in QEMU else data and bss will be pushed onto FLASH or ROM 

    .bss: {*(.bss)*(COMMON)} > RAM

COMMON is another tag used for Global variables and hence all private unitiliazed variables (.bss)
and all the shared unitilialized variables (COMMON) are put into same block

    PROVIDE(_stack_top = ORIGIN(RAM) + LENGTH(RAM));

the compiled code and global variables start at the bottom, that is defined in the linker and grows upwards
the stack grows downwards, to ensure it doesnt overwrite anything we need to define the _stack_top
PROVIDE : GNU linker command that creates a variable which can be used by .c and .S files
ORIGIN : returns the starting address
LENGTH : returns length
both defined in MEMORY segment

## BOOTLOADER
boot.S is a C Runtime Initilialization (CRT) file
For larger systems GRUB or U-Boot is done where it must initiliaze DDR RAM Clocks, 
read file systems from memory, load kernal and hand control to OS

```asm
.section .text.boot, "ax"
.global _start

_start:
    la sp, _stack_top
    call main

1: j 1b
```

.section .text.boot : manual tag to put all instructions into .text.boot section
this allows the linked.ld to push the instructions onoto start of RAM
.global _start defines a globally visible label _start for linker to pull the commands
_start is a lable that designates the official entry point to entire program
"ax": marks the .text.boot as allocatable and executable

commands:
la sp, _stack_top : loads address of _stack_top, which is the variable in linker.ld
which specifies the top of the 128MB RAM onto the stack pointer - sp
as the stack is ready, the processor needs to run the c code
call main : calls tehe main() function placed by the linker, it also leaves the current address
onto the stack so that control can return to bootloader after program completion
1: j 1b = trap the CPU in an infinite loop after completion
add label 1 and jump backwards to label1 ie 1b, creating an infinite loop safety if main returns
without this the control will move onto .rodata which are non executable (here the ML weights)

## NEW Makefile
For RV32IM

```Makefile
CC = riscv64-unknown-elf-gcc
CFLAGS = -march=rv32im -mabi=ilp32 -mcmodel=medany -ffreestanding -O2
LDFLAGS = -nostdlib -T linker.ld

ml_app.elf: boot.S main.c ml_engine.c
	$(CC) $(CFLAGS) $(LDFLAGS) boot.S main.c ml_engine.c -o ml_app.elf

clean:
		rm -f ml_app.elf
```

CC: cross compiler
-march=rv32im : Forcing the riscv64 to compile in rv32im
-mabi=ilp32 : sets app bin interface, guarantees int, long and pointer are 32 bit wide
-mcmodel=medany : instructs the compiler to use PC-relative addressing
-ffreestanding : No OS
-O2 : Level 2 performance optimization

-nostdlib : no auto linkong of std libc library
-T linker.ld : Forces linker to use custom memory map

## QEMU command

` qemu-system-riscv32 -machine virt -bios none -kernel ml_app.elf -nographic `

-machine virt : virtual motherboard with RAM start at 0x80000000
-bios none : no boot firmware
-kernal ml_app.elf : Loads binary to V RAM
-nographic : no output as new window


# STAGE 2

