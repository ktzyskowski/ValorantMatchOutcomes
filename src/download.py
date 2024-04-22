import bs4
import pandas as pd
import requests

from bs4 import BeautifulSoup, Tag
from pandas import DataFrame

url_prefix = 'https://www.vlr.gg'

event_urls = [
    # 2023
    "https://www.vlr.gg/event/matches/1189/?group=completed&series_id=all",  # Americas
    "https://www.vlr.gg/event/matches/1190/?group=completed&series_id=all",  # EMEA
    "https://www.vlr.gg/event/matches/1191/?group=completed&series_id=all",  # Pacific

    # 2024
    "https://www.vlr.gg/event/matches/1923/?series_id=all&group=completed",  # Americas Kickoff
    "https://www.vlr.gg/event/matches/1924/?series_id=all&group=completed",  # Pacific Kickoff
    "https://www.vlr.gg/event/matches/1925/?series_id=all&group=completed",  # EMEA Kickoff
    "https://www.vlr.gg/event/matches/1926/?series_id=all&group=completed",  # China Kickoff
    "https://www.vlr.gg/event/matches/1998/?series_id=all&group=completed",  # EMEA
    "https://www.vlr.gg/event/matches/2002/?series_id=all&group=completed",  # Pacific
    "https://www.vlr.gg/event/matches/2004/?series_id=all&group=completed",  # Americas
    "https://www.vlr.gg/event/matches/2006/?series_id=all&group=completed",  # China
]


def extract_match_urls(event_url: str) -> list[str]:
    """Download the event page from the given URL, and extract match URLs contained within.

    :param event_url: the URL to the event page on VLR.gg.
    :return: the list of URLs for every match in the event on VLR.gg.
    """
    res = requests.get(event_url)
    if res.status_code != 200:
        return []
    soup = BeautifulSoup(res.text, features='html.parser')
    anchor_tags = soup.find_all('a', {'class': 'match-item'})
    hrefs = [url_prefix + anchor_tag['href'] for anchor_tag in anchor_tags]
    return hrefs


def extract_match_data(match_url: str):
    res = requests.get(match_url)
    if res.status_code != 200:
        return []
    soup = BeautifulSoup(res.text, features='html.parser')
    map_tags = [tag for tag in soup.find_all('div', {'class': 'vm-stats-game'}) if tag['data-game-id'] != 'all']
    dfs = [extract_map_data(tag) for tag in map_tags]
    return pd.concat(dfs)


def extract_map_data(map_tag: Tag) -> DataFrame:
    try:
        score0, score1 = map(lambda tag: int(tag.text), map_tag.find_all('div', {'class': 'score'}))
        rows = []
        team0_row_tags, team1_row_tags = [tbody_tag.find_all('tr') for tbody_tag in map_tag.find_all('tbody')]
        for row_tags, won in [(team0_row_tags, score0 > score1), (team1_row_tags, score1 > score0)]:
            for row_tag in row_tags:
                data_tags = row_tag.find_all('td')
                row = {
                    'player': data_tags[0].text.split()[0],
                    'map': map_tag.find('div', {'class': 'map'}).text.split()[0].lower(),
                    'agent': data_tags[1].find('img')['title'].lower(),
                    'rating': float(data_tags[2].text.split()[0]),
                    'acs': int(data_tags[3].text.split()[0]),
                    'kills': int(data_tags[4].text.replace('/', '').split()[0]),
                    'deaths': int(data_tags[5].text.replace('/', '').split()[0]),
                    'assists': int(data_tags[6].text.replace('/', '').split()[0]),
                    'kast': float(data_tags[8].text.replace('%', '').split()[0]) / 100,
                    'adr': int(data_tags[9].text.split()[0]),
                    'hs': float(data_tags[10].text.replace('%', '').split()[0]) / 100,
                    'fk': int(data_tags[11].text.split()[0]),
                    'fd': int(data_tags[12].text.split()[0]),
                    'won': int(won)
                }
                rows.append(row)
        return DataFrame(rows)
    except Exception:
        return DataFrame()


if __name__ == '__main__':
    match_urls = [match_url for event_url in event_urls for match_url in extract_match_urls(event_url)]
    print(len(match_urls))
    exit(0)
    dfs = [extract_match_data(match_url) for match_url in match_urls]
    df = pd.concat(dfs)
    df.to_csv('data.csv')
    print(df.shape)
