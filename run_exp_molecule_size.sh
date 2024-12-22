echo -n > ./data/artificial_molecule.txt
for i in 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 35 40; do
    echo -e "$i\n0.5\n1\n3\n24" | python ./data/gen_with_group.py
done
echo -e "1\n3" | python -m src.script.output_find_group
