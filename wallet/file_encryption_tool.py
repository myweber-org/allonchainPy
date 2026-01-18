import os
import sys

class XORCipher:
    def __init__(self, key: str):
        self.key = key.encode('utf-8')
    
    def _process(self, data: bytes) -> bytes:
        key_length = len(self.key)
        return bytes([data[i] ^ self.key[i % key_length] for i in range(len(data))])
    
    def encrypt_file(self, input_path: str, output_path: str):
        self._process_file(input_path, output_path)
    
    def decrypt_file(self, input_path: str, output_path: str):
        self._process_file(input_path, output_path)
    
    def _process_file(self, input_path: str, output_path: str):
        if not os.path.exists(input_path):
            print(f"Error: Input file '{input_path}' not found.")
            sys.exit(1)
        
        with open(input_path, 'rb') as f_in:
            data = f_in.read()
        
        processed_data = self._process(data)
        
        with open(output_path, 'wb') as f_out:
            f_out.write(processed_data)
        
        print(f"Operation completed. Output saved to '{output_path}'.")

def main():
    if len(sys.argv) != 5:
        print("Usage: python file_encryption_tool.py <encrypt|decrypt> <key> <input_file> <output_file>")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    key = sys.argv[2]
    input_file = sys.argv[3]
    output_file = sys.argv[4]
    
    cipher = XORCipher(key)
    
    if mode == 'encrypt':
        cipher.encrypt_file(input_file, output_file)
    elif mode == 'decrypt':
        cipher.decrypt_file(input_file, output_file)
    else:
        print("Error: Mode must be 'encrypt' or 'decrypt'.")
        sys.exit(1)

if __name__ == "__main__":
    main()