# TP4 Information Retrieval

Link deployment: [PWS](https://juan-maxwell-tp4ir.pbp.cs.ui.ac.id)

## Anggota
- Juan Maxwell Tanaya - 2206820352
- Tengku Laras Malahayati - 2206081641

## Dataset
Dataset yang kami gunakan diambil dari HuggingFace pada link [berikut](https://huggingface.co/datasets/mteb/cqadupstack-programmers), yang terlihat merupakan hasil processing dari [sini](https://ir-datasets.com/beir.html#beir/cqadupstack/programmers). Terdapat 3 buah set, `corpus` yang merupakan list of documents, `queries` yang merupakan list of queries, dan `default` yang merupakan qrels.

## How to Run

Untuk menjalankannya dengan mudah, bisa menggunakan `Docker`
```bash
$ docker compose up -d
```

Untuk menjalankannya tanpa `Docker`, diperlukan minimal `Java` versi `11` lalu jalankan command berikut
```bash
$ pip install -r requirements.txt
$ python3 main.py
```
