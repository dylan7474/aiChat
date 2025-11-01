# Compiler and flags
CC = gcc
CFLAGS = -Wall -g -std=c99
LDFLAGS = -lcurl -ljson-c

# Target executable name
TARGET = aichat

# Source file
SRC = aichat.c

# Phony targets
.PHONY: all clean install

# Default target: build the executable
all: $(TARGET)

# Rule to link the object file into the final executable
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)
	@echo "$(TARGET) has been compiled successfully."

# Rule to clean up build files
clean:
	@echo "Cleaning up build files..."
	rm -f $(TARGET)

# Rule to install the executable to /usr/local/bin
install: $(TARGET)
	@echo "Installing $(TARGET) to /usr/local/bin/..."
	sudo cp $(TARGET) /usr/local/bin/$(TARGET)
	@echo "Installation complete. You can now run '$(TARGET)' from anywhere."
