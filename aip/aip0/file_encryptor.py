
import sys
import os

def xor_cipher(data, key):
    return bytes([b ^ key for b in data])

def encrypt_file(input_path, output_path, key):
    try:
        with open(input_path, 'rb') as f:
            data = f.read()
        encrypted = xor_cipher(data, key)
        with open(output_path, 'wb') as f:
            f.write(encrypted)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    if len(sys.argv) != 4:
        print("Usage: python file_encryptor.py <input> <output> <key>")
        print("Key must be an integer between 0 and 255")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        key = int(sys.argv[3])
        if not 0 <= key <= 255:
            raise ValueError
    except ValueError:
        print("Key must be an integer between 0 and 255")
        sys.exit(1)
    
    if not os.path.exists(input_file):
        print(f"Input file '{input_file}' not found")
        sys.exit(1)
    
    if encrypt_file(input_file, output_file, key):
        print(f"File encrypted successfully: {output_file}")
    else:
        print("Encryption failed")

if __name__ == "__main__":
    main()