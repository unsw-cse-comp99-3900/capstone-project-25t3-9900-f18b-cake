from argparse import ArgumentParser
import json
import html
import re
from thefuzz import fuzz
from functools import reduce

class WHOBookOfTumourClient:
    def __init__(self, file_path: str) -> None:
        with open(file_path, 'r') as f:
            self.books = json.load(f)
    
    def __get_book_by_id(self, id: int) -> object:
        for book in self.books:
            if book['bookid'] == id:
                return book
        return None
    
    def __get_book_by_title(self, title: str) -> object:
        for book in self.books:
            if title.lower() in book['bookTitle'].lower():
                return book
        return None

    def __get_chapters_by_level(self, chapters: list, level: int) -> list:
        return [chapter for chapter in chapters if chapter['chapterLevel'] == level]
    
    def __get_leaf_chapter_by_id(self, chapters: list, id: int) -> object:
        result = [chapter for chapter in self.__get_chapters_by_level(chapters, 4) if chapter['chapterId'] == id]
        if len(result) == 0:
            return None
        return result[0]
    
    def __get_leaf_chapter_by_title(self, chapters : list, title: str) -> object:
        result = [chapter for chapter in chapters if chapter['chapterTitle'] == title]
        if len(result) == 0:
            return None
        return result[0]
    
    def __search_chapter_by_substring(self, target, substring):
        result = []
        search_space = self.books
        
        if target != '':
            book_titles = [book['bookTitle'] for book in self.books]
            matched_book_title = sorted(zip(book_titles, [fuzz.ratio(target, title) for title in book_titles]), key=lambda x: x[1], reverse=True)[0][0]
            search_space = [self.__get_book_by_title(matched_book_title)]
        
        for book in search_space:
            lv4_chapters = reduce(lambda x, y :x + y, [self.__get_chapters_by_level(section['chapters'], 4) for section in book['sections']])
            for chapter in lv4_chapters:
                if substring.lower() in chapter['chapterTitle'].lower():
                    item = {
                        'bookId': book['bookid'],
                        'bookTitle': book['bookTitle'],
                        'chapterId': chapter['chapterId'],
                        'chapterTitle': chapter['chapterTitle'],
                    }
                    result.append(item)
            
        # result.sort(key=lambda x: x['similarity'], reverse=True)

        return result

    def __clean(self, string: str) -> str:
        string = string\
            .replace('&nbsp;', ' ')\
            .replace('\n', ' ')\
            .replace('\r', ' ')\
            .replace('\\u2013', '-')\
            .replace('<sup>', '^')\
            .replace('</sup>', '')\
            .strip()
        string = html.unescape(string)
        string = re.sub(r' +', ' ', string)
        string = re.sub(r'<a.*?a>', '', string)
        return string
    
    def __pretty_print_content(self, key: str, value: str) -> None:
        value = self.__clean(value)
        red = '\033[91m'
        green = '\033[92m'
        blue = '\033[94m'
        bold = '\033[1m'
        italics = '\033[3m'
        underline = '\033[4m'
        end = '\033[0m'

        print(f'{underline}{blue}{bold}{key}{end}:\n')
        value = value.replace('<em>', italics+bold).replace('</em>', end)

        if key in [
            'Definition', 'ICD-O coding', 'ICD-11 coding', 
            'Related terminology', 'Localization',
            'Epidemiology', 'Etiology', 'Cytology', 'Staging', 'Essential and desirable diagnostic criteria'
        ]:
            value = re.sub('</?p ?>', '', value)
            if key == 'Related terminology':
                value = value.replace('Acceptable', f'{green}Acceptable{end}')
                value = value.replace('Not recommended', f'{red}Not recommended{end}')

            print(value + '\n')
        
        if key == 'Subtype(s)':
            value = re.sub('</?p ?>', '', value)
            for v in value.split('; '):
                print(v)
            print()
        
        if key in [
            'Clinical features', 'Pathogenesis', 'Macroscopic appearance', 'Histopathology', 
            'Diagnostic molecular pathology', 'Prognosis and prediction'
        ]:
            for section in re.findall(r'<p>(.*?)</p>', value):
                if len(section.split(' ')) <= 10:
                    print(f'  > {bold}{section}{end}\n')
                else:
                    print(section + '\n')
    
    def __deconstruct_content(self, key: str, value: str) -> any:
        value = self.__clean(value)
        
        if key in [
            'Definition', 'ICD-O coding', 'ICD-11 coding', 
            'Related terminology', 'Localization'
        ]:
            value = re.sub('</?[emp]+ ?>', '', value)
        
        if key == 'Subtype(s)':
            value = re.sub('</?[emp]+ ?>', '', value)
            if 'None' in value:
                value = []
            else:
                value = value.split('; ')
        
        if key in [
            'Clinical features', 'Epidemiology', 'Etiology', 'Pathogenesis',
            'Macroscopic appearance', 'Histopathology', 'Cytology', 'Diagnostic molecular pathology',
            'Essential and desirable diagnostic criteria', 'Staging', 'Prognosis and prediction'
        ]:
            value = [re.sub('</?em ?>', '', v) for v in re.findall(r'<p>(.*?)</p>', value)]

        return value
    
    def __pretty_print_book(self, book):
        """
                'id': book['bookid'],
                'bookTitle': book['bookTitle'],
                'series': book['series'],
                'revision': book['revision'],
                'sections': content

        :param book: _description_
        :type book: _type_
        """ 
        blue = '\033[94m'
        bold = '\033[1m'
        italics = '\033[3m'
        end = '\033[0m'

        print("---------- Book with matching ID ----------\n")
        print(f"{bold}{italics}ID{end}: {book['id']}\n")
        print(f"{bold}{italics}Title{end}: {book['bookTitle']}\n")
        print(f"{bold}{italics}Series{end}: {book['series']}\n")
        print(f"{bold}{italics}Revision{end}: {book['revision']}\n")
        print(f"{bold}{italics}Sections{end}:\n")

        for section in book['sections']:
            print(f"  {bold}{section}{end}: ")
            for chapter in book['sections'][section]:
                print(f"    {blue}{chapter}{end} --chapterId {book['sections'][section][chapter]['chapterId']}")
            
            print()

    
    def get_all_books(
        self,
        verbose: bool = False
    ):
        blue = '\033[94m'
        bold = '\033[1m'
        end = '\033[0m'

        if verbose:
            for book in self.books:
                print(f"{bold}{blue}{book['bookTitle']}{end} --bookId {book['bookid']}\n")

        return [
           {
            'bookId': book['bookid'],
            'bookTitle': book['bookTitle']
           } for book in self.books
        ]
    
    
    def get_book(
        self, 
        bookId: int = None, 
        title: str = None,
        verbose: bool = False
    ):
        """
        Get high level info on a book
        Provide id (requires exact match) or a title (allows partial matching)

        :param bookId: id of book, used for exact matching, defaults to None
        :type bookId: int, optional
        :param title: title of book, defaults to None
        :type title: string, optional
        """

        if bookId:
            book = self.__get_book_by_id(bookId)

        if title:
            book = self.__get_book_by_title(title)
        
        if book:
            content = {}
            for section in book['sections']:
                content[section['chapterTitle']] = {}
                chapters = self.__get_chapters_by_level(section['chapters'], 4)
                if chapters:
                    for chapter in chapters:
                        info = {
                            'chapterId': chapter['chapterId'],
                            'content': self.get_book_chapter(chapterId=chapter['chapterId'], bookId=bookId, clean=True)['content']
                        }
                        content[section['chapterTitle']][chapter['chapterTitle']] = info

            
            result = {
                'id': book['bookid'],
                'bookTitle': book['bookTitle'],
                'series': book['series'],
                'revision': book['revision'],
                'sections': content
            }

            if verbose:
                self.__pretty_print_book(result)

            return result
        
        return {}
    
    def get_book_chapter(
        self, 
        bookId: int = None, 
        chapterId: int = None, 
        bookTitle: str = None, 
        chapterTitle: str = None,
        verbose: bool = False,
        clean: bool = False,
    ):
        """

        Either provide the bookId + chapterId or provide the bookTitle + chapterTitle

        :param bookId: id of the book, defaults to None
        :type bookId: int, optional
        :param chapterId: id of the chapter, defaults to None
        :type chapterId: int, optional
        :param bookTitle: title of the book, defaults to None
        :type bookTitle: str, optional
        :param chapterTitle: title of the chapter, defaults to None
        :type chapterTitle: str, optional
        """

        if bookId and chapterId:
            book = self.__get_book_by_id(bookId)
                      
        if bookTitle and chapterTitle:
            book = self.__get_book_by_title(bookTitle)
        
        for section in book['sections']:
            if section['chapterLevel'] == 1:
                if chapterId:
                    chapter = self.__get_leaf_chapter_by_id(section['chapters'], chapterId)
                
                if chapterTitle:
                    chapter = self.__get_leaf_chapter_by_title(section['chapters'], chapterTitle)

                if chapter:
                    result = {
                        'bookId': book['bookid'],
                        'bookTitle': book['bookTitle'],
                        'chapterId': chapter['chapterId'],
                        'chapterTitle': chapter['chapterTitle'],
                        'content': {},
                        'attachments': []
                    }
                    if verbose:
                        print(f"--------- Book: {book['bookTitle']} ----------\n")
                        print(f"--------- Chapter: {chapter['chapterTitle']} ----------\n")
                    
                    for item in chapter['content']:
                        key = item['headingTitle']
                        value = ''.join([contribution['headingText'] for contribution in item['contributions']])
                        
                        if verbose:
                            self.__pretty_print_content(key, value)
                        
                        if clean:
                            result['content'][key] = self.__deconstruct_content(key, value)
                        else:
                            result['content'][key] = value
                    
                    for item in chapter['attachment']:
                        attachment = {
                            'title': '',
                            'type': item['type'],
                            'content': ''
                        }
                        if item['type'] == 'Figure':
                            attachment['title'] = item['legend']
                            attachment['content'] = item['imageUrl']
                        elif item['type'] == 'Table':
                            attachment['title'] = re.sub(r'<.*?>', '', item['title'])
                            attachment['content'] = item['tablehtml']
                        
                        result['attachments'].append(attachment)
                    
                    return result
        
        return {}
    
    def __highlight_substring(self, substring: str, string: str) -> str:
        highlight_green = '\033[92m'
        highlight_end = '\033[0m'

        start = string.lower().find(substring.lower())
        end = start+len(substring)
        
        string = string[:start] + highlight_green + string[start:]
        string = string[:end+len(highlight_green)] + highlight_end + string[end+len(highlight_green):]
        
        return string
    
    def search(
        self, 
        book: str,
        substring: str,
        verbose: bool = False
    ) -> dict:
        result = self.__search_chapter_by_substring(book, substring)
        if verbose:
            print('---- Search Result ----\n')
            for r in result:
                to_print = f"{r['bookTitle']} >> {r['chapterTitle']}"
                
                to_print = self.__highlight_substring(substring, to_print)
                
                print(to_print)
                print(f"pipenv run python3 WHOBookOfTumourClient.py -v --bookId {r['bookId']} --chapterId {r['chapterId']}\n")
        return result


if __name__ == '__main__':
    parser = ArgumentParser(
        prog = 'WHOBookOfTumourClient', 
        description="""
            Load contents of the parsed WHO Book of Tumour (parsed on 16/02/2023). 
            No arguments will get you all books.
            Just specifying book ID or title will give you information on the book and its chapters (title/id only).
            Also specify chapter ID or title to get detailed chapter content.
        """
    )
    parser.add_argument('--search', dest='search_term', type=str, metavar='Ependymoma', required=False, help='Search for a chapter by disease term')
    parser.add_argument('--bookId', dest='bookId', type=int, metavar=44, required=False, help='Book ID')
    parser.add_argument('--bookTitle', dest='bookTitle', metavar='"Paediatric Tumours"', required=False, help='Book Title')
    parser.add_argument('--chapterId', dest='chapterId', type=int, metavar=94, required=False, help='Chapter ID (must also specify bookId)')
    parser.add_argument('--chapterTitle', dest='chapterTitle', metavar='"Ganglioglioma"', required=False, help='Chapter Title (must also specify bookTitle)')
    parser.add_argument('-c', '--clean', dest='clean', required=False, action='store_true', help='Clean up chapter content and print in json format')
    parser.add_argument('-v', '--verbose', dest='verbose', required=False, action='store_true', help='Print chapter content in a humanly readable format')
    args = parser.parse_args()

    result = None
    data_file = 'api/data/who_book.json'
    inspector = WHOBookOfTumourClient(data_file)

    if not args.search_term and not args.bookId and not args.bookTitle and not args.chapterId and not args.chapterTitle:
        result = inspector.get_all_books(verbose=args.verbose)
    
    if args.search_term:
        result = inspector.search(book='', substring=args.search_term, verbose=args.verbose)
    
    if (args.bookId or args.bookTitle) and (not args.chapterId and not args.chapterTitle):
        if args.bookId:
            result = inspector.get_book(bookId=args.bookId, verbose=args.verbose)
        elif args.bookTitle:
            result = inspector.get_book(title=args.bookTitle, verbose=args.verbose)
    
    if (args.bookId and args.chapterId) or (args.bookTitle and args.chapterTitle):
        if args.bookId and args.chapterId:
            result = inspector.get_book_chapter(
                bookId=args.bookId, 
                chapterId=args.chapterId, 
                verbose=args.verbose,
                clean=args.clean
            )
        elif args.bookTitle and args.chapterTitle:
            result = inspector.get_book_chapter(
                bookTitle=args.bookTitle, 
                chapterTitle=args.chapterTitle, 
                verbose=args.verbose,
                clean=args.clean
            )

    if not args.verbose:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    exit()
    