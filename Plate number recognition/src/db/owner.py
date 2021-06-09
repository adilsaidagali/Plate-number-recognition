class Owner:
    def __init__(self, first_name, middle_name, last_name, sex, birthday):
        self.first_name = first_name
        self.middle_name = middle_name
        self.last_name = last_name
        self.sex = sex
        self.birthday = birthday

    def __str__(self):
        return f'{self.last_name} {self.first_name} {self.middle_name} [{self.sex}] ({self.birthday})'
