"""The three search tools. Each researcher in the graph owns exactly one of them."""
from ddgs import DDGS
from langchain_core.tools import tool

from . import config
from .arabic import normalize_arabic
from .data import Stores


def build_tools(stores: Stores) -> list:
    @tool
    def quran_search_tool(query: str) -> str:
        """Search a vector store of the full Holy Quran (Arabic text, all 114 surahs, 6,236 ayahs),
        grouped into passages of up to 5 consecutive ayahs.
        Use it for any question about Quranic verses, stories of the prophets (e.g. Musa, Ibrahim, Yusuf),
        Islamic teachings, rulings, or topics mentioned in the Quran.
        Input: a specific search query, preferably in Arabic (e.g. 'قصة البقرة مع قوم موسى' or 'الصبر عند المصيبة');
        English also works but Arabic gives better matches.
        Returns: the most relevant passages, each with its surah name and surah:ayah reference."""
        results = stores.quran.similarity_search(normalize_arabic(query), k=config.QURAN_TOP_K)
        return "\n\n".join(
            f"[{i}] Surah {d.metadata.get('surah_name_en', '')} ({d.metadata.get('surah_name_ar', '')}), "
            f"{d.metadata.get('reference', '?')}\n{d.page_content}"
            for i, d in enumerate(results, start=1)
        )

    @tool
    def hadith_search(query: str) -> str:
        """Search Sahih al-Bukhari hadiths by meaning (semantic search), in Arabic or English.
        Use this when the user asks about a hadith, or a saying/action of the Prophet
        Muhammad (peace be upon him), or wants a hadith to complement a Quranic answer.
        Input: a short description of the topic or the hadith wording, in Arabic or English
        (e.g. 'إنما الأعمال بالنيات', 'بر الوالدين', 'actions depend on intentions',
        'kindness to parents'). Exact wording is not required.
        Returns: the most relevant hadiths with their hadith numbers, each in both Arabic and English."""
        # Normalize the same way the Arabic hadiths were normalized before indexing (no-op for English).
        q = normalize_arabic(query)

        # Each hadith is indexed twice (ar + en), so over-fetch and dedupe by hadith number.
        results = stores.hadith.similarity_search(q, k=config.HADITH_TOP_K * 4)
        seen, top = set(), []
        for d in results:
            n = d.metadata["hadithnumber"]
            if n not in seen:
                seen.add(n)
                top.append(stores.hadith_by_number[n])
            if len(top) == config.HADITH_TOP_K:
                break

        if not top:
            return "No matching hadiths found."
        return "\n\n".join(
            f"Hadith #{h['hadithnumber']} (Sahih al-Bukhari):\n"
            f"AR: {h['text_ar'] or '(Arabic text unavailable)'}\n"
            f"EN: {h['text_en']}"
            for h in top
        )

    @tool
    def duck_duck_search(query: str) -> str:
        """Search the web with DuckDuckGo for information that is NOT in the Quran text itself.
        Use it for tafsir (scholarly explanations of verses), hadith, English translations of ayahs,
        the historical context or reasons of revelation (asbab al-nuzul), biographies of scholars or companions,
        current Islamic events and dates (e.g. 'Ramadan 2026 start date'), or anything the Quran search could not answer.
        Do NOT use it to find or quote Quranic verses themselves -- always use quran_search_tool for that,
        since it returns the exact Arabic text with surah:ayah references.
        Input: a concise search query, in English or Arabic, as you would type into a search engine
        (e.g. 'tafsir Ibn Kathir Al-Baqarah 67' or 'تفسير آية الكرسي').
        Returns: the top web results, each with a title, URL, and short snippet."""
        try:
            results = DDGS().text(query, max_results=config.WEB_TOP_K)
        except Exception as e:
            return f"Search failed: {e}"
        return "\n\n".join(
            f"[{i}] {r['title']} ({r['href']})\n{r['body']}"
            for i, r in enumerate(results, start=1)
        ) or "No web results."

    return [hadith_search, duck_duck_search, quran_search_tool]
