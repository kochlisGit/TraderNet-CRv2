import requests
from abc import ABC, abstractmethod
from urllib.parse import urlencode
from database.network.network import DatasetDownloader


class CoinAPIDownloader(DatasetDownloader, ABC):
    def __init__(self, verbose: bool):
        super().__init__(date_column_name=self._get_date_column_name(), verbose=verbose)

        self._api_key_list = [
            '70E10174-E29D-449F-9F2E-6E8362931DD9',
            '27E5E40C-7A6B-45EB-A5C8-8311B049A741',
            '8F6252DE-0AD7-478F-91C7-141141E8BE8B',
            '3B49210E-100B-4F8D-9011-2BA5D38274BA',
            'BF6BF46F-B44B-416E-9656-2D2AAFBC058B',
            'B21A98A2-C953-4C73-84CF-CFFB6F712200',
            '51667E99-7686-4496-B23D-6DA54F7E37AE',
            '0921F87B-BF55-4B78-B8B0-E023B4D7A2E2',
            '3F9E3251-029C-457A-9ADA-7F21A440AAF9',
            '41EBEA2D-1A4B-4654-8A41-186639B9AB9F',
            '6B93AEC2-910C-4064-80FB-91AED487AB97',
            '83049379-23DE-4CB0-8299-7137BB836D48',
            'B08FCA1F-F454-4C34-AC01-42F16354BCBC',
            '12E5D72C-25A6-4ED6-8384-7C291EC43768',
            '4F287859-5A00-47EF-AC91-8A2629F8C6A1',
            '3744F705-2C4A-406C-AA96-EB1B557A84EF',
            '3F77D500-457E-4A96-9CE1-1DEF3FC7033B',
            '455C2228-0D6F-4B62-8336-4BAA24C1A46E',
            '7E37E058-670C-4ED6-B7BE-DC00F309D9FF',
            '0F517C3D-162C-4C5E-AE18-544B201C9BC0'
        ]

    @property
    def api_key_list(self) -> list[str]:
        return self._api_key_list

    @abstractmethod
    def _get_date_column_name(self) -> str:
        pass

    @abstractmethod
    def _get_request_params(self) -> dict[str, str]:
        pass

    @staticmethod
    def _encode_request_url(base_url: str, request_params: dict, api_key: str) -> str:
        request_params['apikey'] = api_key
        encoded_params = urlencode(request_params)
        return f'{base_url}?{encoded_params}'

    def _get_response(self, base_url: str, request_params: dict) -> requests.Response or None:
        for api_key in self._api_key_list:
            if self._verbose:
                print(f'Using apikey: {api_key}')

            encoded_request_url = self._encode_request_url(
                base_url=base_url,
                request_params=request_params,
                api_key=api_key
            )
            response = requests.get(encoded_request_url)

            if self._verbose:
                print(f'Response Status: {response.status_code} - {response.reason}')

            if response.status_code == 200:
                return response
        return None
