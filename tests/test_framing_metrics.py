"""
Unit tests for the label-free framing evaluation metrics.

Each metric is checked against hand-built profiles whose correct answer can be
computed by hand, so the test says what the number MEANS, not just that the
function runs.
"""
import pytest

from models.framing.metrics import (
    attribution_score,
    distinctiveness,
    evaluate_extractor,
    generic_term_rate,
    split_by_media,
    tokenize,
)


class TestTokenize:
    def test_lowercases_and_keeps_words_of_three_or_more_chars(self):
        assert tokenize("The KPU Announced It") == ["the", "kpu", "announced"]

    def test_punctuation_and_digits_are_dropped(self):
        assert tokenize("58.59 percent, officially!") == ["percent", "officially"]


class TestDistinctiveness:
    def test_identical_profiles_score_one(self):
        profiles = {
            "tempo": [("election", 1.0), ("cabinet", 1.0)],
            "antaranews": [("election", 1.0), ("cabinet", 1.0)],
        }
        assert distinctiveness(profiles) == pytest.approx(1.0)

    def test_disjoint_profiles_score_zero(self):
        profiles = {
            "tempo": [("corruption", 1.0), ("investigation", 1.0)],
            "antaranews": [("cooperation", 1.0), ("development", 1.0)],
        }
        assert distinctiveness(profiles) == pytest.approx(0.0)

    def test_half_shared_profiles_score_the_jaccard_value(self):
        # intersection = {election}, union = {election, corruption, cooperation}
        profiles = {
            "tempo": [("election", 1.0), ("corruption", 1.0)],
            "antaranews": [("election", 1.0), ("cooperation", 1.0)],
        }
        assert distinctiveness(profiles) == pytest.approx(1 / 3)

    def test_single_source_has_no_pairs_to_compare(self):
        assert distinctiveness({"tempo": [("election", 1.0)]}) == 0.0

    def test_averages_over_all_pairs(self):
        profiles = {
            "a": [("x", 1.0)],
            "b": [("x", 1.0)],
            "c": [("y", 1.0)],
        }
        # pairs: a-b = 1.0, a-c = 0.0, b-c = 0.0
        assert distinctiveness(profiles) == pytest.approx(1 / 3)


class TestGenericTermRate:
    def test_keywords_that_are_corpus_filler_are_counted(self):
        corpus = ["indonesia indonesia indonesia election election news"]
        profiles = {"tempo": [("indonesia", 1.0), ("election", 1.0)]}

        assert generic_term_rate(profiles, corpus, top_n=2) == pytest.approx(1.0)

    def test_rare_keywords_are_not_generic(self):
        corpus = ["indonesia indonesia indonesia election election news"]
        profiles = {"tempo": [("dynasties", 1.0), ("corruption", 1.0)]}

        assert generic_term_rate(profiles, corpus, top_n=2) == pytest.approx(0.0)

    def test_mixed_profile_gives_a_partial_rate(self):
        corpus = ["indonesia indonesia election election news analysts"]
        profiles = {"tempo": [("indonesia", 1.0), ("dynasties", 1.0)]}

        assert generic_term_rate(profiles, corpus, top_n=2) == pytest.approx(0.5)

    def test_empty_profiles_do_not_divide_by_zero(self):
        assert generic_term_rate({}, ["some text here"]) == 0.0


class TestSplitByMedia:
    def test_every_source_is_represented_in_both_halves(self):
        articles = {"tempo": [f"t{i}" for i in range(10)], "antaranews": [f"a{i}" for i in range(20)]}

        train, test = split_by_media(articles, test_size=0.25)

        assert set(train) == {"tempo", "antaranews"}
        assert {media for _, media in test} == {"tempo", "antaranews"}

    def test_split_is_deterministic_for_a_fixed_seed(self):
        articles = {"tempo": [f"t{i}" for i in range(10)]}

        assert split_by_media(articles, seed=1) == split_by_media(articles, seed=1)

    def test_no_article_appears_in_both_halves(self):
        articles = {"tempo": [f"t{i}" for i in range(12)]}

        train, test = split_by_media(articles)

        assert not set(train["tempo"]) & {text for text, _ in test}

    def test_all_articles_are_accounted_for(self):
        articles = {"tempo": [f"t{i}" for i in range(12)], "abc_news": [f"a{i}" for i in range(8)]}

        train, test = split_by_media(articles)

        assert sum(len(v) for v in train.values()) + len(test) == 20


class TestAttributionScore:
    @staticmethod
    def perfect_profiles(texts_by_media, top_k):
        """An oracle extractor: each outlet gets the marker word only it uses."""
        return {media: [(media.split("_")[0], 1.0)] for media in texts_by_media}

    @staticmethod
    def useless_profiles(texts_by_media, top_k):
        """An extractor that gives every outlet the same generic word."""
        return {media: [("indonesia", 1.0)] for media in texts_by_media}

    def test_perfectly_separating_keywords_score_full_accuracy(self):
        articles = {
            "tempo": [f"tempo reported story {i}" for i in range(8)],
            "abc_news": [f"abc covered story {i}" for i in range(8)],
        }

        result = attribution_score(articles, self.perfect_profiles)

        assert result["accuracy"] == pytest.approx(1.0)
        assert result["macro_f1"] == pytest.approx(1.0)

    def test_non_discriminating_keywords_collapse_to_one_class(self):
        articles = {
            "tempo": [f"indonesia story {i}" for i in range(8)],
            "abc_news": [f"indonesia story {i}" for i in range(8)],
        }

        result = attribution_score(articles, self.useless_profiles)

        # Everything is predicted as a single label, so macro-F1 is far below 1
        assert result["macro_f1"] < 0.5

    def test_majority_baseline_reflects_class_imbalance(self):
        articles = {
            "tempo": [f"tempo story {i}" for i in range(4)],
            "antaranews": [f"antaranews story {i}" for i in range(16)],
        }

        result = attribution_score(articles, self.perfect_profiles)

        assert result["majority_baseline"] == pytest.approx(0.8)

    def test_result_is_deterministic(self):
        articles = {
            "tempo": [f"tempo story {i}" for i in range(8)],
            "abc_news": [f"abc story {i}" for i in range(8)],
        }

        assert attribution_score(articles, self.perfect_profiles) == attribution_score(
            articles, self.perfect_profiles
        )

    def test_n_test_matches_the_held_out_count(self):
        articles = {"tempo": [f"tempo story {i}" for i in range(8)]}

        result = attribution_score(articles, self.perfect_profiles, test_size=0.25)

        assert result["n_test"] == 2


class TestEvaluateExtractor:
    def test_returns_every_metric(self):
        articles = {
            "tempo": [f"tempo corruption story {i}" for i in range(8)],
            "abc_news": [f"abc democracy story {i}" for i in range(8)],
        }

        result = evaluate_extractor(articles, TestAttributionScore.perfect_profiles)

        for key in ("distinctiveness", "generic_term_rate", "accuracy", "macro_f1", "profiles"):
            assert key in result


class TestProductionExtractors:
    """The adapters must work on the real analyzers, not just fixtures."""

    @pytest.fixture
    def corpus(self):
        return {
            "tempo": [
                "Corruption investigation targets regional officials in a widening probe.",
                "Investigation into military dynasties raises concerns among observers.",
            ],
            "antaranews": [
                "The government announced cooperation on regional development programs.",
                "Development cooperation with ASEAN partners expands this year.",
            ],
        }

    def test_wordcount_adapter_returns_weighted_pairs(self, corpus):
        from models.framing.metrics import wordcount_profiles

        profiles = wordcount_profiles(corpus, top_k=5)

        assert set(profiles) == {"tempo", "antaranews"}
        for keywords in profiles.values():
            assert all(isinstance(term, str) and isinstance(weight, float) for term, weight in keywords)

    def test_tfidf_adapter_returns_weighted_pairs(self, corpus):
        from models.framing.metrics import tfidf_profiles

        profiles = tfidf_profiles(corpus, top_k=5)

        assert set(profiles) == {"tempo", "antaranews"}
        for keywords in profiles.values():
            assert keywords, "TF-IDF returned no keywords"
            assert all(isinstance(term, str) for term, _ in keywords)

    def test_wordcount_keeps_ties_deterministic(self, corpus):
        from models.framing.metrics import wordcount_profiles

        assert wordcount_profiles(corpus, top_k=8) == wordcount_profiles(corpus, top_k=8)
