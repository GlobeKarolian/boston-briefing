(async () => {
  const byId = (s) => document.getElementById(s);
  const yearEl = byId('year');
  yearEl.textContent = new Date().getFullYear();

  const feedUrl = './feed.xml';
  try {
    const res = await fetch(feedUrl, { cache: 'no-store' });
    if (!res.ok) throw new Error(`Feed fetch failed: ${res.status}`);
    const xml = new window.DOMParser().parseFromString(await res.text(), 'text/xml');

    const item = xml.querySelector('channel > item');
    if (!item) throw new Error('No items in feed');

    const title = (item.querySelector('title')?.textContent || 'Latest Episode').trim();
    byId('ep-title').textContent = title;

    const enclosure = item.querySelector('enclosure');
    const audioUrl = enclosure?.getAttribute('url') || item.querySelector('link')?.textContent;
    if (audioUrl) {
      byId('player').src = audioUrl;
      const dl = byId('download');
      dl.href = audioUrl;
      dl.download = audioUrl.split('/').pop();
    }

    const m = title.match(/(\d{4}-\d{2}-\d{2})/);
    const shownotesHref = m ? `./shownotes/${m[1]}.html` : './shownotes/';
    byId('shownotes').href = shownotesHref;
  } catch (err) {
    console.error(err);
    byId('ep-title').textContent = 'Latest episode unavailable';
  }
})();