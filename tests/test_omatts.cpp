// omatts unit tests — pure logic only, no models, no audio hardware.
//
// Includes omatts.cpp with PTT_SHARED_LIB defined, which drops main() and the
// daemon/playback code but keeps everything else, so statics are reachable.
//
// Build & run:  make test   (cmake target `omatts-test`)

#define PTT_SHARED_LIB
#include "../omatts.cpp"

#include <cassert>
#include <iostream>

static int g_failed = 0, g_total = 0;

#define CHECK(cond) do { \
    ++g_total; \
    if (!(cond)) { \
        ++g_failed; \
        std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << "  " #cond "\n"; \
    } \
} while (0)

#define CHECK_EQ(a, b) do { \
    ++g_total; \
    auto _va = (a); auto _vb = (b); \
    if (!(_va == _vb)) { \
        ++g_failed; \
        std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << "  " #a " == " #b \
                  "  (got [" << _va << "] want [" << _vb << "])\n"; \
    } \
} while (0)

static bool approx(float a, float b, float eps = 1e-4f) { return std::abs(a - b) < eps; }

// ── text chunking ───────────────────────────────────────────────────────────

static void test_split_pauses() {
    auto plain = omatts::split_pauses("Hello world");
    CHECK_EQ(plain.size(), 1u);
    CHECK_EQ(plain[0].text, "Hello world");
    CHECK(approx(plain[0].pause, 0.0f));
    CHECK(approx(plain[0].volume, 1.0f));

    auto p = omatts::split_pauses("A [[pause 1]] B");
    CHECK_EQ(p.size(), 2u);
    CHECK_EQ(p[0].text, "A ");
    CHECK(approx(p[0].pause, 1.0f));
    CHECK_EQ(p[1].text, " B");
    CHECK(approx(p[1].pause, 0.0f));

    auto def = omatts::split_pauses("A [[pause]] B");  // no number -> 0.5
    CHECK_EQ(def.size(), 2u);
    CHECK(approx(def[0].pause, 0.5f));

    auto clamp = omatts::split_pauses("[[pause 99]]");
    CHECK_EQ(clamp.size(), 2u);
    CHECK(approx(clamp[0].pause, 10.0f));  // clamped to [0, 10]

    auto v = omatts::split_pauses("[[volume 99]] soft [[volume 0.01]] loud [[volume]] back");
    CHECK_EQ(v[0].text, "");            // text before a tag keeps the previous volume
    CHECK(approx(v[1].volume, 2.0f));   // clamped to [0.1, 2]
    CHECK_EQ(v[1].text, " soft ");
    CHECK(approx(v[2].volume, 0.1f));
    CHECK(approx(v[3].volume, 1.0f));   // [[volume]] with no number resets
    CHECK_EQ(v[3].text, " back");

    auto bad = omatts::split_pauses("[[volume abc]] x");  // garbage -> reset
    CHECK(approx(bad[0].volume, 1.0f));

    auto unterm = omatts::split_pauses("A [[pause 1 B");  // unterminated tag = literal
    CHECK_EQ(unterm.size(), 1u);
    CHECK_EQ(unterm[0].text, "A [[pause 1 B");

    auto neg = omatts::split_pauses("[[pause -5]] x");
    CHECK(approx(neg[0].pause, 0.0f));  // clamped to 0
}

static void test_sentences_with_pauses() {
    // Pause rides on the last chunk of its segment.
    auto c = omatts::sentences_with_pauses("One two. Three four. [[pause 1]] Five.");
    CHECK(!c.empty());
    CHECK(approx(c[c.size() - 2].pause, 1.0f));  // last chunk of the paused segment
    CHECK(approx(c.back().pause, 0.0f));         // "Five." is its own segment
    // All sentences present, in order, joined or split within the token budget.
    std::string all;
    for (auto& ch : c) all += ch.text;
    CHECK(all.find("One two.") != std::string::npos);
    CHECK(all.find("Five.") != std::string::npos);

    // Short sentences merge into one chunk (fewer cold starts = fewer artifacts).
    auto m = omatts::sentences_with_pauses("Hi there. How are you. I am fine.");
    CHECK_EQ(m.size(), 1u);
    CHECK_EQ(m[0].text, "Hi there. How are you. I am fine.");

    // Volume state flows into chunks.
    auto v = omatts::sentences_with_pauses("Soft. [[volume 2]] Loud.");
    CHECK(approx(v.front().volume, 1.0f));
    CHECK(approx(v.back().volume, 2.0f));

    // A lone pause produces a silent chunk, not a crash or dropped audio.
    auto lone = omatts::sentences_with_pauses("[[pause 2]]");
    CHECK_EQ(lone.size(), 1u);
    CHECK_EQ(lone[0].text, "");
    CHECK(approx(lone[0].pause, 2.0f));

    // Empty / whitespace-only input yields no chunks at all.
    CHECK(omatts::sentences_with_pauses("").empty());
    CHECK(omatts::sentences_with_pauses("   \n\t ").empty());
}

static void test_chunk_budget() {
    // Five 10-word sentences: the 50-token budget flushes before the 5th.
    // (Verified: chunk 0 = 4 sentences / 40 words, chunk 1 = the rest.)
    std::string s;
    for (int i = 0; i < 5; i++) s += "one two three four five six seven eight nine ten. ";
    auto c = omatts::sentences_with_pauses(s + "[[pause 1]]");
    CHECK_EQ(c.size(), 2u);
    CHECK_EQ(omatts::count_words(c[0].text), 40);
    CHECK_EQ(omatts::count_words(c[1].text), 10);
    CHECK(approx(c[0].pause, 0.0f));
    CHECK(approx(c[1].pause, 1.0f));  // pause rides the LAST chunk after a split
}

static void test_split_sentences() {
    auto s = omatts::split_sentences("First one. Second one! Third?");
    CHECK_EQ(s.size(), 3u);

    // Abbreviations don't end a sentence.
    auto ab = omatts::split_sentences("Mr. Smith met Dr. Who. They talked.");
    CHECK_EQ(ab.size(), 2u);

    // Ellipsis and decimals don't split; "?!" stays with the sentence.
    auto el = omatts::split_sentences("Wait... what?!");
    CHECK_EQ(el.size(), 1u);
    CHECK_EQ(el[0], "Wait... what?!");
    auto dec = omatts::split_sentences("Pi is 3.14 exactly.");
    CHECK_EQ(dec.size(), 1u);

    // No trailing punctuation: the tail still comes back.
    auto tail = omatts::split_sentences("One. Two");
    CHECK_EQ(tail.size(), 2u);
    CHECK_EQ(tail[1], "Two");

    CHECK(omatts::split_sentences("   ").empty());
}

// ── text preparation ────────────────────────────────────────────────────────

static void test_prepare_text() {
    auto r = omatts::prepare_text("hello world", -1, false, false);
    CHECK_EQ(r.first, "Hello world.");  // capitalized + final period
    CHECK_EQ(r.second, 5);  // <=4 words -> 5 extra frames

    // The eos_extra auto rule branches exactly at 4/5 words.
    CHECK_EQ(omatts::prepare_text("one two three four", -1, false, false).second, 5);
    CHECK_EQ(omatts::prepare_text("one two three four five", -1, false, false).second, 3);

    auto short_p = omatts::prepare_text("hi", -1, true, false);
    CHECK_EQ(short_p.first, "        Hi.");  // 8-space pad for <5 words

    auto nospace = omatts::prepare_text("   \n\t  ", -1, true, false);
    CHECK_EQ(nospace.first, "");

    auto semi = omatts::prepare_text("a; b", -1, false, true);
    CHECK_EQ(semi.first, "A, b.");

    // Quotes stripped, contractions kept.
    auto q = omatts::prepare_text("\"don't stop\" it's", -1, false, false);
    CHECK(q.first.find("on't stop") != std::string::npos);
    CHECK(q.first.find('\"') == std::string::npos);
    CHECK_EQ(q.first.back(), '.');

    // Explicit eos_extra wins.
    auto eos = omatts::prepare_text("hello", 7, false, false);
    CHECK_EQ(eos.second, 7);

    // UTF-8 curly quotes/apostrophes are stripped (the size()>=3 edge loops).
    CHECK_EQ(omatts::prepare_text("\xe2\x80\x9cquoted\xe2\x80\x9d", -1, false, false).first, "Quoted.");
    CHECK_EQ(omatts::prepare_text("\xe2\x80\x98tis\xe2\x80\x99", -1, false, false).first, "Tis.");
}

// ── normalize rules ─────────────────────────────────────────────────────────

static void test_norm_rules() {
    std::istringstream in(
        "# comment\n"
        "([0-9]) ?%\t$1 percent\n"
        "bad line without tab\n"
        "(((\tbroken regex\n"      // regex_error -> skipped with warning
        "\\$\t dollars \n");
    CHECK(omatts::load_norm_rules(in, "test"));      // valid rules win
    CHECK_EQ(omatts::normalize_symbols("50%"), "50 percent");
    CHECK_EQ(omatts::normalize_symbols("$5"), " dollars 5");  // $ expanded, $1 backrefs intact

    // Rules run in order: the next rule sees the previous rule's output.
    std::istringstream order("a\tb\nb\tc\n");
    CHECK(omatts::load_norm_rules(order, "test"));
    CHECK_EQ(omatts::normalize_symbols("a"), "c");

    // normalize_symbols is the pipeline's single choke point: split_sentences
    // must see normalized text.
    std::istringstream pipe("([0-9]) ?%\t$1 percent\n");
    CHECK(omatts::load_norm_rules(pipe, "test"));
    auto s = omatts::split_sentences("50% off. Fine.");
    CHECK_EQ(s.size(), 2u);
    CHECK_EQ(s[0], "50 percent off.");

    std::istringstream none("# only comments\n");
    CHECK(!omatts::load_norm_rules(none, "test"));  // nothing valid -> false
    omatts::g_norm_rules.clear();
    CHECK_EQ(omatts::normalize_symbols("50%"), "50%");  // passthrough when no rules
}

static void test_init_norm_rules() {
    namespace fs = std::filesystem;
    auto tmp = fs::temp_directory_path() / "omatts-test-norm";
    fs::remove_all(tmp);
    fs::create_directories(tmp / "models");
    fs::create_directories(tmp / "models-de");
    {
        std::ofstream f(tmp / "models" / "normalize.txt");
        f << "\\$\t dollars \n";
        std::ofstream g(tmp / "models-de" / "normalize.txt");
        g << "\\$\t Euro \n";
    }
    omatts::init_norm_rules((tmp / "models").string());
    CHECK_EQ(omatts::normalize_symbols("$5"), " dollars 5");
    // A pack switch must not inherit the previous pack's rules.
    omatts::init_norm_rules((tmp / "models-de").string());
    CHECK_EQ(omatts::normalize_symbols("$5"), " Euro 5");
    // A pack with no rules file turns normalization off.
    omatts::init_norm_rules((tmp / "empty-pack").string());
    CHECK(omatts::g_norm_rules.empty());
    CHECK_EQ(omatts::normalize_symbols("$5"), "$5");
    fs::remove_all(tmp);
}

// ── JSON helpers ────────────────────────────────────────────────────────────

static void test_json() {
    CHECK_EQ(omatts::json_get_string("{\"input\":\"hello\"}", "input"), "hello");
    CHECK_EQ(omatts::json_get_string("{\"input\":\"a\\\"b\\\\c\\nd\"}", "input"), "a\"b\\c\nd");
    CHECK_EQ(omatts::json_get_string("{\"input\":\"caf\\u00e9\"}", "input"), "caf\xc3\xa9");
    // Surrogate pair -> U+1F600 = F0 9F 98 80
    CHECK_EQ(omatts::json_get_string("{\"input\":\"\\uD83D\\uDE00\"}", "input"),
             std::string("\xf0\x9f\x98\x80", 4));
    CHECK_EQ(omatts::json_get_string("{\"input\":\"x\"}", "missing"), "");
    CHECK_EQ(omatts::json_get_string("not json", "input"), "");
    // Value after the right key (not an earlier occurrence of the key text).
    CHECK_EQ(omatts::json_get_string("{\"speed\":2,\"voice\":\"sam\"}", "voice"), "sam");

    CHECK(approx(omatts::json_get_float("{\"speed\":1.5}", "speed", 1.0f), 1.5f));
    CHECK(approx(omatts::json_get_float("{\"speed\":2}", "speed", 1.0f), 2.0f));
    CHECK(approx(omatts::json_get_float("{}", "speed", 3.0f), 3.0f));

    // Round-trip: escape then parse back.
    std::string tricky = "He said \"hi\"\n\ttab\\slash \xc3\xa9";
    CHECK_EQ(omatts::json_get_string("{\"text\":\"" + omatts::json_escape(tricky) + "\"}", "text"), tricky);
}

static void test_fnv1a_hex() {
    CHECK_EQ(omatts::fnv1a_hex(""), "cbf29ce484222325");           // FNV-1a-64 offset basis
    CHECK_EQ(omatts::fnv1a_hex("a"), "af63dc4c8601ec8c");
}

// ── audio helpers ───────────────────────────────────────────────────────────

static void test_wav_encode() {
    std::vector<float> samples = {0.1f, -0.2f, 0.3f};
    auto w = omatts::TTSServer::wav_encode(samples.data(), samples.size(), 24000);
    CHECK_EQ(w.size(), 44u + 12u);
    CHECK_EQ(std::string(reinterpret_cast<const char*>(&w[0]), 4), "RIFF");
    CHECK_EQ(std::string(reinterpret_cast<const char*>(&w[8]), 4), "WAVE");
    uint16_t fmt; memcpy(&fmt, &w[20], 2);
    CHECK_EQ(fmt, 3);  // IEEE float
    uint32_t rate; memcpy(&rate, &w[24], 4);
    CHECK_EQ(rate, 24000u);
    uint32_t dsz; memcpy(&dsz, &w[40], 4);
    CHECK_EQ(dsz, 12u);

    // Zero-length edge (the base for the CLI's 0xFFFFFFFF stream header).
    auto h = omatts::TTSServer::wav_encode(&samples[0], 0, 24000);
    uint32_t riff; memcpy(&riff, &h[4], 4);
    CHECK_EQ(riff, 36u);  // file size with a zero-length data chunk
    uint32_t dsz0; memcpy(&dsz0, &h[40], 4);
    CHECK_EQ(dsz0, 0u);
}

static void test_resample() {
    std::vector<float> one = {1, 2, 3, 4};
    CHECK_EQ(omatts::resample(one, 24000, 24000).size(), 4u);  // identity
    auto up = omatts::resample(std::vector<float>(2400, 0.5f), 24000, 48000);
    CHECK_EQ(up.size(), 4800u);
    auto down = omatts::resample(std::vector<float>(4800, 0.5f), 48000, 24000);
    CHECK_EQ(down.size(), 2400u);
    // DC stays DC: one aggregate deviation check, not 2400 identical asserts.
    float maxdev = 0;
    for (float f : down) maxdev = std::max(maxdev, std::abs(f - 0.5f));
    CHECK(maxdev < 1e-3f);
    // Empty in, empty out.
    CHECK(omatts::resample({}, 24000, 48000).empty());
}

static void test_apply_volume() {
    std::vector<float> s = {0.25f, 1.5f, -1.5f};
    omatts::TTSServer::apply_volume(s, 2.0f);
    CHECK(approx(s[0], 0.5f));
    CHECK(approx(s[1], 1.0f));   // clamped
    CHECK(approx(s[2], -1.0f));  // clamped
    std::vector<float> u = {0.5f};
    omatts::TTSServer::apply_volume(u, 1.0f);
    CHECK(approx(u[0], 0.5f));   // 1.0 is a no-op
}

static void test_misc() {
    CHECK(is_audio_ext(".wav") && is_audio_ext(".mp3") && is_audio_ext(".flac"));
    CHECK(!is_audio_ext(".txt") && !is_audio_ext(".WAV"));
    CHECK_EQ(omatts::count_words("  one two  three "), 3);
    CHECK_EQ(omatts::count_words(""), 0);
}

// ── regression: fixed bugs stay fixed ──────────────────────────────────────

static void test_regressions() {
    namespace fs = std::filesystem;
    // Cache key collision: same basename in different subdirs must not share entries.
    auto a = omatts::cache::get_cache_path("/v", "a/alba.wav", "emb", "/m");
    auto b = omatts::cache::get_cache_path("/v", "b/alba.wav", "emb", "/m");
    CHECK(a != b);
    CHECK(a.find("/v/.cache/") == 0);
    // A different model set must not share entries either.
    CHECK(a != omatts::cache::get_cache_path("/v", "a/alba.wav", "emb", "/m2"));

    // Corrupt .emb files are rejected, not parsed into garbage.
    auto tmp = fs::temp_directory_path() / "omatts-test-cache";
    fs::create_directories(tmp);
    {
        std::ofstream f(tmp / "trunc.emb", std::ios::binary);
        f.write("EMB1", 4);
        f.put('\x02');  // ndims=2, then nothing — truncated
    }
    {
        std::ofstream f(tmp / "huge.emb", std::ios::binary);  // absurd shape
        f.write("EMB1", 4);
        int32_t ndims = 1;
        int64_t shape = int64_t(1) << 40;
        f.write(reinterpret_cast<const char*>(&ndims), 4);
        f.write(reinterpret_cast<const char*>(&shape), 8);
    }
    {
        std::ofstream f(tmp / "ok.emb", std::ios::binary);
        std::vector<float> data = {1, 2, 3};
        CHECK(omatts::cache::save_embedding((tmp / "ok.emb").string(), {1, 3}, data));
    }
    std::vector<int64_t> shape; std::vector<float> data;
    CHECK(!omatts::cache::load_embedding((tmp / "trunc.emb").string(), shape, data));
    CHECK(!omatts::cache::load_embedding((tmp / "huge.emb").string(), shape, data));
    CHECK(!omatts::cache::load_embedding((tmp / "missing.emb").string(), shape, data));
    CHECK(omatts::cache::load_embedding((tmp / "ok.emb").string(), shape, data));
    CHECK(shape == std::vector<int64_t>({1, 3}) && data == std::vector<float>({1, 2, 3}));

    // Cache validity: invalid when the voice is newer, missing, or the cache is stale.
    auto voice = tmp / "v.wav", cached = tmp / "v.emb";
    { std::ofstream(voice) << "a"; std::ofstream(cached) << "b"; }
    CHECK(omatts::cache::is_cache_valid(voice.string(), cached.string()));
    fs::last_write_time(cached, fs::file_time_type::clock::now() - std::chrono::hours(1));
    CHECK(!omatts::cache::is_cache_valid(voice.string(), cached.string()));
    CHECK(!omatts::cache::is_cache_valid(voice.string(), (tmp / "missing.emb").string()));
    fs::remove_all(tmp);
}

// ── config pack switching (temp dirs) ───────────────────────────────────────

static void test_config_pack() {
    namespace fs = std::filesystem;
    auto tmp = fs::temp_directory_path() / "omatts-test-config";
    fs::remove_all(tmp);
    fs::create_directories(tmp / "models");
    fs::create_directories(tmp / "models-de");

    omatts::Config cfg;
    cfg.models_dir = (tmp / "models").string();
    // Missing pack throws with an actionable message; present pack switches.
    bool threw = false;
    try { cfg.find_models_dir("fr"); } catch (const std::exception&) { threw = true; }
    CHECK(threw);
    CHECK_EQ(cfg.find_models_dir("de"), (tmp / "models-de").string());
    CHECK_EQ(cfg.find_models_dir(""), cfg.models_dir);  // empty = current pack

    // model_config.txt drives the per-pack flags; load_models_dir picks it up.
    {
        std::ofstream f(tmp / "models-de" / "model_config.txt");
        f << "language=de\nremove_semicolons=1\n";
        std::ofstream t(tmp / "models-de" / "tokenizer.model");
        t << "x";
    }
    cfg.use_language_pack("de");
    CHECK_EQ(cfg.language, "de");
    CHECK(cfg.remove_semicolons);
    CHECK_EQ(cfg.models_dir, (tmp / "models-de").string());

    fs::remove_all(tmp);
}

// ── -v voice tag resolution ─────────────────────────────────────────────────

static void test_resolve_voice_tag() {
    namespace fs = std::filesystem;
    auto tmp = fs::temp_directory_path() / "omatts-test-voicetag";
    fs::remove_all(tmp);
    fs::create_directories(tmp / "models");
    fs::create_directories(tmp / "voices" / "de");
    fs::create_directories(tmp / "voices" / "en");
    auto touch = [&](const fs::path& p) { std::ofstream f(p); f << "x"; };
    touch(tmp / "voices" / "de" / "anna.ogg");
    touch(tmp / "voices" / "de" / "klaus.ogg");
    touch(tmp / "voices" / "de" / "zzz-last.ogg");
    touch(tmp / "voices" / "en" / "alba.ogg");

    omatts::Config cfg;
    cfg.models_dir = (tmp / "models").string();
    cfg.voices_dir = (tmp / "voices").string();

    // Absolute path and existing file are paths, not tags.
    std::string abs = (tmp / "voices" / "de" / "klaus.ogg").string();
    std::string v = abs;
    CHECK_EQ(resolve_voice_tag(cfg, v), "");
    CHECK_EQ(v, abs);

    // Explicit tag/name: the tag is the prefix.
    v = "de/klaus";
    CHECK_EQ(resolve_voice_tag(cfg, v), "de");
    CHECK_EQ(v, "de/klaus");

    // A bare voice present in exactly one language folder expands to tag/name.
    v = "alba";
    CHECK_EQ(resolve_voice_tag(cfg, v), "en");
    CHECK_EQ(v, "en/alba");

    // A bare language tag = the alphabetically first voice in that folder.
    v = "de";
    CHECK_EQ(resolve_voice_tag(cfg, v), "de");
    CHECK_EQ(v, "de/anna");

    // Unknown name: no tag; the regular not-found error fires later.
    v = "nobody";
    CHECK_EQ(resolve_voice_tag(cfg, v), "");
    CHECK_EQ(v, "nobody");

    fs::remove_all(tmp);
}

int main() {
    test_split_pauses();
    test_sentences_with_pauses();
    test_split_sentences();
    test_prepare_text();
    test_norm_rules();
    test_json();
    test_fnv1a_hex();
    test_wav_encode();
    test_resample();
    test_apply_volume();
    test_misc();
    test_regressions();
    test_config_pack();
    test_chunk_budget();
    test_init_norm_rules();
    test_resolve_voice_tag();

    if (g_failed) {
        std::cerr << g_failed << "/" << g_total << " checks FAILED\n";
        return 1;
    }
    std::cout << g_total << " checks passed\n";
    return 0;
}
