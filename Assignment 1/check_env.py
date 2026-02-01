try:
    import sklearn
    from sklearn.datasets import fetch_20newsgroups
    import seaborn
    import matplotlib
    import wordcloud
    print("Imports success")
except Exception as e:
    print(f"Import failed: {e}")
