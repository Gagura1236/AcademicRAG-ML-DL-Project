from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ACADEMIC_STOP_WORDS = {
    "model", "models", "used", "using", "use", "tasks", "task", "paper", "papers", 
    "method", "methods", "approach", "approaches", "result", "results", "based", 
    "proposed", "propose", "study", "studies", "show", "shows", "shown", "data",
    "dataset", "datasets", "set", "sets", "performance", "state", "art", "different",
    "work", "works", "new", "problem", "problems", "analysis", "experiments", "experiment",
    "table", "figure", "fig", "time", "learning", "training", "trained", "test", "testing",
    "evaluation", "evaluate", "evaluated", "compare", "compared", "comparison", "include",
    "includes", "including", "feature", "features", "input", "output", "value", "values",
    "number", "important", "improve", "improved", "improvement", "algorithm", "algorithms",
    "information", "process", "processes", "application", "applications", "research", "researchers",
    "author", "authors", "et", "al", "section", "sections", "following", "given", "known",
    "large", "small", "high", "low", "increase", "decrease", "networks", "network", "neural",
    "layer", "layers", "function", "functions", "equation", "equations", "parameters", "parameter",
    "image", "images", "text", "texts", "word", "words", "language", "languages",
    "available", "provide", "provides", "provided", "make", "makes", "made", "however", "therefore",
    "size", "sizes", "deep", "simple", "complex", "good", "bad", "better", "best", "accuracy", "accurate"
}

CUSTOM_STOP_WORDS = list(ENGLISH_STOP_WORDS.union(ACADEMIC_STOP_WORDS))
