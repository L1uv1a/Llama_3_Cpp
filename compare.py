import struct

def read_floats_from_file(filename):
    """Read 32-bit floating-point numbers from a binary file."""
    with open(filename, "rb") as file:
        data = file.read()
        # Unpack the data into a list of floats
        floats = struct.unpack(f"{len(data) // 4}f", data)
    return floats

def write_floats_to_file(filename, floats):
    """Write a list of 32-bit floating-point numbers to a binary file."""
    with open(filename, "wb") as file:
        # Pack the floats into binary format and write to the file
        file.write(struct.pack(f"{len(floats)}f", *floats))

def subtract_floats(file1, file2, output_file=None):
    """Read two binary files, subtract their floats, and optionally save the result."""
    floats1 = read_floats_from_file(file1)
    floats2 = read_floats_from_file(file2)
    
    if len(floats1) != len(floats2):
        raise ValueError("The files do not have the same number of floating-point numbers.")
    
    # Subtract corresponding floats
    result = [abs(abs(a) - abs(b)) for a, b in zip(floats1, floats2)]

    result = [abs(a - b) for a, b in zip(floats1, floats2)]
    # Optionally write the result to a new binary file
    if output_file:
        write_floats_to_file(output_file, result)
    
    return result

def find_minimum_in_file(file):
    """Find the minimum value in a binary file."""
    floats = read_floats_from_file(file)
    return min(floats, key=abs)

def find_maximum_in_file(file):
    """Find the minimum value in a binary file."""
    floats = read_floats_from_file(file)
    return max(floats, key=abs)

def calculate_mean(file):
    """Calculate the mean of numbers in a binary file."""
    floats = read_floats_from_file(file)
    # Calculate the mean as sum of floats divided by the count
    mean_value = sum(floats) / len(floats)
    return mean_value

def find_nearest_to_zero_nonzero(differences):
    """Find the value closest to 0 but not equal to 0."""
    non_zero_differences = [x for x in differences if x != 0]
    if not non_zero_differences:
        raise ValueError("All differences are zero or the list is empty.")
    nearest_to_zero = min(non_zero_differences, key=abs)
    return nearest_to_zero

# Example usage
file1 = "logits.bin"
file2 = "output_fp32.bin"
output_file = "result.bin"

# Perform the subtraction and save the result
result = subtract_floats(file1, file2, output_file=output_file)
min_value1 = find_minimum_in_file(file1)
min_value2 = find_minimum_in_file(file2)
max_value1 = find_maximum_in_file(file1)
max_value2 = find_maximum_in_file(file2)
mean_value = calculate_mean(file1)
mean_difference_value = sum(result)/len(result)
max_diff = max(result)
min_diff = min(result)
near_diff_non_0 = find_nearest_to_zero_nonzero(result)
print("Min Subtraction result:", min_diff)
print("Max Subtraction result:", max_diff)
print("Nearest diff non zero result:", near_diff_non_0)
print("Minimum value in C++ file:", min_value1)
print("Max value in C++ file 1:", max_value1)
print("Mean value:", mean_value)
print("Mean difference:", mean_difference_value)
print("Relative Error", mean_difference_value/mean_value)