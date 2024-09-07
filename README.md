# connectome

Instructions for when cuda breaks

```{bash}
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm
```

Coses a fer
[-] Agafar la posició més superficial
[x] Fer el canvi B i R
[x] Crear les cel·les de Voronoi amb centres en les R7 o R8
[-] Crear neurones acumuladores per veure si aprenen els números
[x] Mirar si hi ha alguna neurona que sempre ho endevina
[x] Posar millor els plans de la retina
[x] Mirar com evoluciona la imatge en les diferents capes del sistema visual
[x] Veure si es degrada randomitzant els edges

Idees
[x] Quins dos colors poden diferenciar millor les mosques
[x] Reshuffle pesos i no entrenar
[x] Treure neurones: entrenant o no entrenant
[x] Deixar fixes els pesos i mirar si sap alguna cosa: reservoir computing
[x] Treure neurones (sinapses) tipus a tipus
[x] Fer que el model entrenat per fer una cosa, en faci una altra
[x] Assegurar-se que les dues maneres de testejar són equivalents
[x] Analitzar les head direction neurons


[x] Refined matrix sense entrenar
[x] Refined matrix with restricted training (\theta  \in [0, 1])
[x] Head direction with single stripe
[x] Mirar la relació entre radi/distància/color i posició al manifold
[x] Augmentar el número de capes quan no surt
[x] mirar manifolds sobre el test set
[x] provar amb menys punts
[x] amb xarxa entrenada amb un color, veure si també funciona per altres colors
[-] transfer learning: entrenar a la vegada punts i símbols
[-] mnist
[x] Neuron subselection for decision making
[ ] Manifolds: see how neural representations evolve with training
[ ] Manifolds: check shapes trained only on edges

Paper:
- weber
- distingir coses
- manifolds: hi ha uns certs patrons que s'activen per unes raons i creen 

what can flies do?

query per la posició de les neurones: https://codex.flywire.ai/app/view_3d?query=cell_type+%7Bin%7D+R1-6%2CR7%2CR8+%7Band%7D+side+%7Bequal%7D+right&action=random_sample&color_by=Type&mode=3D+only