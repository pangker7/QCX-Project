for i in 2 3 4 5 6; do
    echo -e "10\n0.5\n1\n$i\n24" | python ./src/gen_with_group.py
    echo -e "0\n$i" | python ./src/output_find_group.py
    # Add your commands here
done