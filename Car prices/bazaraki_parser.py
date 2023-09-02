import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import json
import copy
import os


class GetCarsBazaraki():

    def __init__(self, brands, models):

        self.brands = brands
        self.models = models

    def get_ads(self):

        # lists for hrefs
        self.hrefs = list()
        #lists for dataframe
        name = list()
        price = list()
        price_currency = list()
        place = list()
        posted_on = list()
        chars_list = list()
        views = list()
        description = list()
        ref = list()
        ref_bad = list()
        response = list()
        response_bad = list()

        for brand in self.brands:
            for model in self.models:
                for page_number in range(10):
                    url = f'https://www.bazaraki.com/car-motorbikes-boats-and-parts/cars-trucks-and-vans/{brand}/{model}/?page={page_number}'
                    bazaraki = requests.get(url)
                    soup = BeautifulSoup(bazaraki.text, "html.parser")
                    ads = soup.find_all("a", class_="mask", )
                    for ad in ads:
                        if ad.get("href") not in self.hrefs and ad.get("href")[:4]=='/adv':
                            self.hrefs.append(ad.get("href"))

        for href in self.hrefs:
            url_ad = 'https://www.bazaraki.com'+href
            bazaraki_ad = requests.get(url_ad)
            soup = BeautifulSoup(bazaraki_ad.text, "html.parser")
            try:
                price.append(soup.find("div", class_="announcement-price__cost").find_next().find_next().get('content'))
                price_currency.append(soup.find("div", class_="announcement-price__cost").find_next().get('content'))
                place.append(soup.find("span", itemprop="address").string)
                posted_on.append(soup.find("span", class_="date-meta").string)
                chars = {}
                key_chars = [x.string for x in soup.find_all("span", class_="key-chars")]
                value_chars = [x.string for x in soup.find_all(class_="value-chars")]
                for i in range(len(key_chars)):
                    chars[key_chars[i]] = value_chars[i]
                chars_list.append(chars)
                views.append(soup.find("span", class_="counter-views").string)
                description.append([x.string for x in soup.find("div", class_="js-description").find_all('p')])
                ref.append(url_ad)
                name.append(soup.find("h1", id="ad-title").string)
                response.append(bazaraki_ad)
            except:
                ref_bad.append(url_ad)
                response_bad.append(bazaraki_ad)

        df_success = pd.DataFrame()
        df_failure = pd.DataFrame()
        df_success['name'] = name
        df_success['price'] = price
        df_success['price_currency'] = price_currency
        df_success['place'] = place
        df_success['posted_on'] = posted_on
        df_success['chars_list'] = chars_list
        df_success['views'] = views
        df_success['description'] = description
        df_success['response'] = response
        df_success['ref'] = ref
        df_failure['response'] = response_bad
        df_failure['ref'] = ref_bad

        self.df_success = df_success
        self.df_failure = df_failure

        return self

    def write_to_csv(self, path):
        self.df_success.to_csv(path, index=False)

    # Add representation of the class
    def __repr__(self):
        return f"GetCarsBazaraki(brands={self.brands}, models={self.models})"

def update_dataset(new_df, old_df):
    old = pd.read_csv(old_df)
    new = pd.read_csv(new_df)
    df = pd.concat([old, new], axis='rows').drop_duplicates(subset=['ref'])
    os.remove(old_df)
    os.remove(new_df)
    df.to_csv(old_df, index=False)

def clean_data(path):
    df = pd.read_csv(path)
    df['name'] = df['name'].replace('\n', '', regex=True).apply(lambda x: x.strip())
    df['brand'] = df['name'].apply(lambda x: x.split(' ')[0])
    df['model'] = df['name'].apply(lambda x: ' '.join(x.split(' ')[1:-2]).strip())
    df = df[(df['model'].isin(['Demio', '2']))]
    df['year'] = [eval(x)['Year:'] for x in df['chars_list']]
    df['year'] = [float(x) for x in df['year']]
    df['fuel_type'] = [eval(x)['Fuel type:'] if 'Fuel type:' in x else np.nan for x in df['chars_list']]
    df['fuel_type'] = np.where(df['fuel_type']=='Diesel', 1, 0)
    df['gear_box'] = [eval(x)['Gearbox:'] if 'Gearbox:' in x else np.nan for x in df['chars_list']]
    df['gear_box'] = np.where(df['gear_box']=='Manual', 1, 0)
    df['engine_size'] = [eval(x)['Engine size:'][:-1].replace(',', '.') if 'Engine size:' in x else np.nan for x in df['chars_list']]
    df['engine_size'] = np.where(df['engine_size']=='Electri',0,df['engine_size'])
    df['engine_size'] = df['engine_size'].astype(float)
    df['mileage_km'] = [eval(x)['Mileage (in km):'] if 'Mileage (in km):' in x else np.nan for x in df['chars_list']]
    df['mileage_km'] = [float(x[:-3]) for x in df['mileage_km']]
    df['price_eur'] = df.loc[:,'price'].astype(float)
    df = df[['brand', 'model', 'year', 'mileage_km', 'price_eur', 'fuel_type', 'gear_box', 'engine_size', 'ref']]
    try:
        os.remove(path[:-4]+'_clean'+path[-4:])
    except FileNotFoundError:
        pass
    df.to_csv(path[:-4]+'_clean'+path[-4:], index=False)


def run_updates(brand_list, model_list, path_new, path_old):
    get_car = GetCarsBazaraki(brand_list, model_list)
    get_car.get_ads().write_to_csv(path_new)
    update_dataset(path_new, path_old)
    clean_data(path_old)
