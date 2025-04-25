#!/bin/bash

# Define the directory to process
DIRECTORY="./offers1.csv"  # Change this to your actual directory path

# Check if output file argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No output file specified."
    echo "Usage: $0 <output_filename.csv>"
    exit 1
fi

# Define the output CSV file from command line argument
OUTPUT_FILE="$1"

# Check if directory exists
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: Directory $DIRECTORY does not exist."
    exit 1
fi

# Create an array to store filenames
declare -a filenames

# Create a variable to store the filename with letters
letter_filename=""

# Loop through each file in the directory
for file in "$DIRECTORY"/*; do
    if [ -f "$file" ]; then
        # Extract just the filename without the path
        filename=$(basename "$file")
        
        # Check if filename contains letters
        if [[ "$filename" =~ [a-zA-Z] ]]; then
            letter_filename="$filename"
        else
            filenames+=("$filename")
        fi
    fi
done

# Clear the output file if it exists
> "$OUTPUT_FILE"

# Add the file with letters as the first line if it exists
if [ ! -z "$letter_filename" ]; then
    echo "$letter_filename" > "$OUTPUT_FILE"
fi

# Add the remaining files
for filename in "${filenames[@]}"; do
    echo "$filename" >> "$OUTPUT_FILE"
done

echo "File names have been saved to $OUTPUT_FILE"