Below is a tutorial for Raha and Baran that walks through how to use these two data-error detection and repair systems.

# Raha and Baran — Usage Guide

Raha and Baran are configuration-free error detection and error repair systems. Raha detects errors in data; Baran repairs them. Together they form a two-step pipeline that achieves high precision and recall.

## Installation
Install Raha and Baran with:
```bash
pip3 install raha
```

To uninstall:
```bash
pip3 uninstall raha
```

## Usage
Raha and Baran are straightforward to use. Here are common workflows:

### 1. Benchmarking
If you have a dirty dataset and its clean counterpart and want to benchmark Raha and Baran, see the example code in `raha/benchmark.py`, `raha/detection.py`, and `raha/correction.py`. These files illustrate the benchmarking pipeline.

### 2. Interactive data cleaning
If you have a dirty dataset and want to detect and repair errors interactively, open the Jupyter notebooks in the `raha` folder. They provide a GUI that lets you label and correct data interactively.

Through the GUI you can:
   - **Label data**: Use the labeling widgets to identify errors
   - **Pick strategies**: Review candidate strategies and choose the best cleaning approach
   - **Drill-down analysis**: Analyze data clusters to uncover error sources
   - **Dashboard**: Track overall cleaning progress and effectiveness

Screenshots of the interface:
   ![Data labeling](pictures/ui.png)
   ![Strategies](pictures/ui_strategies.png)
   ![Drill-down](pictures/ui_clusters.png)
   ![Dashboard](pictures/ui_dashboard.png)

## Naming
"Raha" and "Baran" are Persian female names chosen to echo the systems' properties. *Raha* ("freedom" in Persian) reflects the "configuration-free" nature of the error detection system. *Baran* ("rain" in Persian, with the connotation that rain washes everything clean) reflects the role of the error repair system in "cleaning" data.

## Citations
If you would like to cite these systems or the accompanying papers, use the BibTeX entries below.

### Cite Raha
```
@inproceedings{mahdavi2019raha,
  title={Raha: A configuration-free error detection system},
  author={Mahdavi, Mohammad and Abedjan, Ziawasch and Castro Fernandez, Raul and Madden, Samuel and Ouzzani, Mourad and Stonebraker, Michael and Tang, Nan},
  booktitle={Proceedings of the International Conference on Management of Data (SIGMOD)},
  pages={865--882},
  year={2019},
  organization={ACM}
}
```

### Cite Baran
```
@article{mahdavi2020baran,
  title={Baran: Effective error correction via a unified context representation and transfer learning},
  author={Mahdavi, Mohammad and Abedjan, Ziawasch},
  journal={Proceedings of the VLDB Endowment (PVLDB)},
  volume={13},
  number={11},
  pages={1948--1961},
  year={2020},
  publisher={VLDB Endowment}
}
```

More information about the project and its authors:
- [Raha: A configuration-free error detection system](https://dl.acm.org/doi/abs/10.1145/3299869.3324956)
- [Baran: Effective error correction via a unified context representation and transfer learning](https://dl.acm.org/doi/abs/10.14778/3407790.3407801)

If you have further questions or need additional help using Raha and Baran for data cleaning, feel free to reach out.
