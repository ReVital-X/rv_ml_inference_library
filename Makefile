CC = riscv64-unknown-elf-gcc
CFLAGS = -march=rv32im -mabi=ilp32 -mcmodel=medany -ffreestanding -O2
LDFLAGS = -nostdlib -T linker.ld

ml_app.elf: boot.S main.c ml_engine.c
	$(CC) $(CFLAGS) $(LDFLAGS) boot.S main.c ml_engine.c -o ml_app.elf

clean:
		rm -f ml_app.elf