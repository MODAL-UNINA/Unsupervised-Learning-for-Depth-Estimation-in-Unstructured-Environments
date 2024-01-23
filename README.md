# Unsupervised Learning for Depth Estimation in Unstructured Environments
This is a Pytorch implementation of the following paper:

Pian Qi, Fabio Giampaolo, Edoardo Prezioso, Francesco Piccialli, Unsupervised Learning for Depth Estimation in Unstructured Environments. [Paper](link)

## Requirements
python>=3.6

pytorch>=0.4

## Note
Follow the official website of the Mid-Air dataset [here](https://midair.ulg.ac.be/) to download and place the files into the folder named "Dataset".


### Example data folder structure:

```
├── Dataset
│   ├── train
│   │   ├── trajectory_0000
│   │   │   ├─ image_left
│   │   │   │   ├── 00001.png
│   │   │   │   └── ...
│   │   │   ├─ image_right
│   │   │   │   ├── 00001.png
│   │   │   │   └── ...
│   │   ├── ...
│   ├── test
│   │   ├── trajectory_1111
│   │   │   ├─ image_left
│   │   │   │   ├── 00001.png
│   │   │   │   └── ...
│   │   │   ├─ image_right
│   │   │   │   ├── 00001.png
│   │   │   │   └── ...
│   │   ├── ...
```

## Acknowledgments
This work has been designed and developed under the “PON Ricerca e Innovazione 2014-2020”– Dottorati innovativi con caratterizzazione industriale XXXVI Ciclo, Fondo per lo Sviluppo e la Coesione, code DOT1318347, CUP E63D20002530006.
This work was also supported by the following projects: 
- PNRR project FAIR -  Future AI Research (PE00000013), Spoke 3, under the NRRP MUR program funded by the NextGenerationEU;
- G.A.N.D.A.L.F. - Gan Approaches for Non-iiD Aiding Learning in Federations, CUP: E53D23008290006, PNRR - Missione 4 “Istruzione e Ricerca” - Componente C2 Investimento 1.1 “Fondo per il Programma Nazionale di Ricerca e Progetti  di Rilevante Interesse Nazionale (PRIN)”.
