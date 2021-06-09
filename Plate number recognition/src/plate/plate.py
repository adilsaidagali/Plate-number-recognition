class LicensePlateError(ValueError):
    pass


class InappropriateSeriesCharacterError(LicensePlateError):
    pass


class InappropriateNumberCharacterError(LicensePlateError):
    pass


class LicensePlate:
    _series_character_white_list = None
    _number_character_white_list = None

    def __init__(self, series, number, region):
        self.series = series
        self.number = number
        self.region = region

    def __str__(self):
        raise NotImplementedError()

    @staticmethod
    def _get_series_entry_from_character(character: str):
        raise NotImplementedError()

    @staticmethod
    def _get_number_entry_from_character(character: str):
        raise NotImplementedError()

    @staticmethod
    def from_characters(characters):
        raise NotImplementedError()


class RuLicensePlate(LicensePlate):
    _series_character_white_list = ['A',
                                    'B',
                                    'C',
                                    'E',
                                    'H',
                                    'K',
                                    'M',
                                    'O',
                                    'P',
                                    'T',
                                    'X',
                                    'Y',
                                    '8',  # B
                                    '0',  # O
                                    '7',  # T
                                    ]
    _number_character_white_list = ['0',
                                    '1',
                                    '2',
                                    '3',
                                    '4',
                                    '5',
                                    '6',
                                    '7',
                                    '8',
                                    '9',
                                    'O',  # 0
                                    'T',  # 7
                                    'B',  # 8
                                    ]

    def __init__(self, series, number, region):
        super().__init__(series, number, region)

    def __str__(self):
        return f'{self.series[0]}{"".join(self.number)}{"".join(self.series[1:3])}{"".join(self.region[:len(self.region)])}'

    @staticmethod
    def _get_series_entry_from_character(character: str):
        upper_character = character.upper()

        if upper_character not in RuLicensePlate._series_character_white_list:
            raise InappropriateSeriesCharacterError()

        if upper_character == '8':
            return 'B'
        elif upper_character == '0':
            return 'O'
        elif upper_character == '7':
            return 'T'
        else:
            return upper_character

    @staticmethod
    def _get_number_entry_from_character(character: str):
        upper_character = character.upper()

        if upper_character not in RuLicensePlate._number_character_white_list:
            raise InappropriateNumberCharacterError()

        if upper_character == 'B':
            return '8'
        elif upper_character == 'O':
            return '0'
        elif upper_character == 'T':
            return '7'
        else:
            return upper_character

    @staticmethod
    def from_characters(characters):
        series = ''.join([RuLicensePlate._get_series_entry_from_character(characters[0]),
                          RuLicensePlate._get_series_entry_from_character(characters[4]),
                          RuLicensePlate._get_series_entry_from_character(characters[5])])
        number = ''.join([RuLicensePlate._get_number_entry_from_character(characters[1]),
                          RuLicensePlate._get_number_entry_from_character(characters[2]),
                          RuLicensePlate._get_number_entry_from_character(characters[3])])
        region = [RuLicensePlate._get_number_entry_from_character(characters[6]),
                  RuLicensePlate._get_number_entry_from_character(characters[7])]
        if len(characters) == 9:
            region.append(RuLicensePlate._get_number_entry_from_character(characters[8]))
        region = ''.join(region)

        return RuLicensePlate(series, number, region)


class KzLicensePlate(LicensePlate):
    _series_character_white_list = ['A',
                                    'B',
                                    'C',
                                    'D',
                                    'E',
                                    'F',
                                    'G',
                                    'H',
                                    'I',
                                    'J',
                                    'K',
                                    'L',
                                    'M',
                                    'N',
                                    'O',
                                    'P',
                                    'Q',
                                    'R',
                                    'S',
                                    'T',
                                    'U',
                                    'V',
                                    'W',
                                    'X',
                                    'Y',
                                    'Z',
                                    '8',  # B
                                    '1',  # I
                                    '0',  # O
                                    '5',  # S
                                    '7',  # T
                                    '2',  # Z
                                    ]
    _number_character_white_list = ['0',
                                    '1',
                                    '2',
                                    '3',
                                    '4',
                                    '5',
                                    '6',
                                    '7',
                                    '8',
                                    '9',
                                    'O',  # 0
                                    'I',  # 1
                                    'Z',  # 2
                                    'S',  # 5
                                    'T',  # 7
                                    'B',  # 8
                                    ]

    def __init__(self, series, number, region):
        super().__init__(series, number, region)

    def __str__(self):
        return f'{"".join(self.number)}{"".join(self.series)}{"".join(self.region)}'

    @staticmethod
    def _get_series_entry_from_character(character: str):
        upper_character = character.upper()

        if upper_character not in KzLicensePlate._series_character_white_list:
            raise InappropriateSeriesCharacterError()

        if upper_character == '8':
            return 'B'
        elif upper_character == '1':
            return 'I'
        elif upper_character == '0':
            return 'O'
        elif upper_character == '5':
            return 'S'
        elif upper_character == '7':
            return 'T'
        elif upper_character == '2':
            return 'Z'
        else:
            return upper_character

    @staticmethod
    def _get_number_entry_from_character(character: str):
        upper_character = character.upper()

        if upper_character not in KzLicensePlate._number_character_white_list:
            raise InappropriateNumberCharacterError()

        if upper_character == 'O':
            return '0'
        elif upper_character == 'I':
            return '1'
        elif upper_character == 'Z':
            return '2'
        elif upper_character == 'S':
            return '5'
        elif upper_character == 'T':
            return '7'
        elif upper_character == 'B':
            return '8'
        else:
            return upper_character

    @staticmethod
    def from_characters(characters):
        number = ''.join([KzLicensePlate._get_number_entry_from_character(characters[0]),
                          KzLicensePlate._get_number_entry_from_character(characters[1]),
                          KzLicensePlate._get_number_entry_from_character(characters[2])])

        series = [KzLicensePlate._get_series_entry_from_character(characters[3]),
                  KzLicensePlate._get_series_entry_from_character(characters[4])]
        if len(characters) == 8:
            series.append(KzLicensePlate._get_series_entry_from_character(characters[5]))
        series = ''.join(series)

        region = ''.join([KzLicensePlate._get_number_entry_from_character(characters[len(characters) - 2]),
                          KzLicensePlate._get_number_entry_from_character(characters[len(characters) - 1])])

        return KzLicensePlate(series, number, region)
