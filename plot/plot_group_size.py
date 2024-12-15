import matplotlib.pyplot as plt
import os

# Function to parse the file and extract data
def parse_file(file_path):
    x_values = []
    success_prob = []
    valid_prob = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Process every 4 lines
        for i in range(0, len(lines), 4):
            # Skip the first two lines (rubbish)
            # Extract y1 and y2 from the third line
            y1, y2 = lines[i + 2].split()
            y1 = float(y1)
            y2 = float(y2)

            # Extract x from the fourth line
            x = int(lines[i + 3].strip())

            # Append to lists
            x_values.append(x)
            success_prob.append(y1)
            valid_prob.append(y2)

    return x_values, success_prob, valid_prob

# Main function to plot the data
def plot_data(file_path):
    # Parse the file
    x_values, success_prob, valid_prob = parse_file(file_path)

    # Plot success_prob vs x
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x_values, success_prob, marker='o', linestyle='-', color='b', label='Success Probability (y1)')
    plt.xlabel('x')
    plt.ylabel('Success Probability (y1)')
    plt.title('Success Probability vs x')
    plt.legend()
    plt.grid(True)

    # Plot valid_prob vs x
    plt.subplot(1, 2, 2)
    plt.plot(x_values, valid_prob, marker='o', linestyle='-', color='r', label='Valid Probability (y2)')
    plt.xlabel('x')
    plt.ylabel('Valid Probability (y2)')
    plt.title('Valid Probability vs x')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()

# Run the script
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "../output/output_find_carboxyl_0.txt")
    plot_data(input_path)