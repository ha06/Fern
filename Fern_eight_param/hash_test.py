class contact_list():

    def __init__(self, names):
        self.names = names

    def __hash__(self):
        return hash(frozenset(self.names))

    def __eq__(self, other):
        return set(self.names) == set(other.names)


def merge_contact_list(contact):
    return list(set(contact))

if __name__ == '__main__':

    h1 = contact_list("sabbir")
    h2 = contact_list("abbirs")
    h3 = contact_list("total")
    h4 = contact_list("latto")
    c_list = [h1, h2, h3, h4]
    print([each.names for each in merge_contact_list(c_list)])
    if h1 == h2:
        print("we are anagram")