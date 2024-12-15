for i in 2 3 4 5 6; do
    echo -n > ./data/artificial_molecule.txt
    echo -e "10\n0.5\n1\n$i\n24" | python ./data/gen_with_group.py
    echo -e "0\n$i" | python ./data/output_find_group.py
    echo $i >> ./output/output_find_carboxyl_0.txt
done