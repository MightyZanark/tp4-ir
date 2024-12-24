if __name__ == "__main__":
    if not pt.java.started():
        pt.java.init()
    
    cache_dir = os.path.abspath(os.path.join(CUR_DIR, "./dataset"))
    ds = load_dataset("mteb/cqadupstack-programmers", "corpus", cache_dir=cache_dir)
    ds = ds["corpus"].to_pandas()