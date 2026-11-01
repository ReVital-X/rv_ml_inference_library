CC = riscv64-unknown-elf-gcc
CFLAGS = -march=rv32im -mabi=ilp32 -mcmodel=medany -ffreestanding -O2 -Iinclude
LDFLAGS = -nostdlib -T linker.ld

ml_app.elf: src/boot.S main.c src/ml_engine.c
	$(CC) $(CFLAGS) $(LDFLAGS) src/boot.S main.c src/ml_engine.c -o ml_app.elf

test: src/ml_engine.c tests/unity.c tests/test_engine.c 
	gcc -O2 -Iinclude src/ml_engine.c tests/unity.c tests/test_engine.c -o test_runner
	./test_runner

clean:
		rm -f ml_app.elf