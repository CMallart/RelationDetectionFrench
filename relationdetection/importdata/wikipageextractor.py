import re
import bz2
import urllib.parse

"""Wikipedia Extractor:
Extracts and cleans text from Wikipedia database dump and stores output in a
number of files of similar size in a given directory. In each line of these files,
a wikipedia page is stored in JSON format.
(Forked and bettering https://github.com/sjebbara/Wikipedia2Json for our purpose).
"""

class AnnotatedWikiDocument(dict):
    __slots__ = ['id', 'url', "title", 'text', 'annotations', 'categories']

    def __init__(self, **kwargs):
        super(AnnotatedWikiDocument, self).__init__(**kwargs)
        self.id = None
        self.url = None
        self.title = None
        self.text = None
        self.annotations = None
        self.categories = None

    def __str__(self):
        self["id"] = self.id
        self["url"] = self.url
        self["title"] = self.title
        self["text"] = self.text
        self["annotations"] = self.annotations
        self["categories"] = self.categories
        #return ujson.dumps(self) + "\n"
        return(self.text)

def get_wiki_document_url(wiki_document_title, prefix, quote=False):
    if quote:
        title = urllib.parse.quote(
            wiki_document_title.replace(' ', '_').encode('utf-8'))
        title = title.replace('%28', '(')
        title = title.replace('%29', ')')
        title = title.replace('%3A', ':')
        title = title.replace('%22', '"')
        title = title.replace('%27', "'")
    else:
        title = wiki_document_title.replace(' ', '_')

    return prefix + title[0].upper() + title[1:]


class WikiExtractor():
    __garbage_tags = (
        'ref', 'gallery', 'timeline', 'noinclude', 'pre', 'table', 'tr', 'td', 'ul', 'li', 'ol', 'dl', 'dt', 'dd',
        'menu', 'dir', 'math')

    __wrapper_tags = (
        'nowiki', 'cite', 'source', 'hiero', 'div', 'font', 'span', 'strong', 'strike', 'blockquote', 'tt', 'var',
        'sup', 'sub', 'big', 'small', 'center', 'h1', 'h2', 'h3', 'em', 'b', 'i', 'u', 'a', 's', 'p', "dfn")

    __single_tags = ('references', "ref",'img', 'br', 'hr', 'li', 'dt', 'dd', 'noinclude')#ref

    __project_namespaces = (
        'wikipedia', 'mediawiki', 'wikiquote', 'wikibooks', 'wikisource', 'wiktionary', 'wikispecies', 'wikinews',
        'wikiversity', 'commons', 'wikicities', 'wikispot')

    __garbage_link_prefixes = (
        'image', 'category', 'file', 'http', 'https', 'simple', 'meta', 'wikipedia', "media", 'template', 'portal',
        'user', 'wikt', 'wikihow', "help", "user talk", "special", "s", "b", "v", "q", "?", "fichier")

    __garbage_page_prefixes = (
        'Image:', 'Category:', 'File:', 'Wikipedia:', 'Template:', 'Portal:', 'User:', "Help:", "Book:", "Draft:",
        "Module:", "TimedText:", "MediaWiki:", "Catégorie:", "Wikipédia:", "Modèle:", "Fichier:", "Projet:")

    __garbage_template_name = ( 
        "Article détaillé", "Lien web", "Autres projets","Portail","Références","Référence souhaitée","À compléter", "Confusion", "à sourcer", "Catégorie détaillée",
        "Article général","Section vide ou incomplète","Article connexe", "Refnec","À vérifier", "À wikifier","À délister", "source",   "sources à lier","Sources à lier",
        "Notoriété","admissibilité","Admissibilité", "Sources secondaires", "Une pomme est un fruit"
        "Voir homonymes","Voir homonymie", "Homonyme","Homonymie","homonymes", "homophone", "Voir homophones", "Voir paronyme", "Patronymie","toponymie"
        "Ébauche","ébauche", "loupe", "Loupe",  "Exemple",  "Légende", "Taxobox","sous-titre","Prononciation","Note",
        "Boîte déroulante/début", "Boîte déroulante/fin", "Arbre généalogique","Démographie","Sommaire","Documentation",
        "Blason-ville-fr", "Blason-ville", "Unicode",
        "Écrit", "Ouvrage","pistes", "Album", "album",
        "Tableau",  "coord", "#tag:ref",'redirect', "colonnes", 
        "climat",     "Palette", "images",  'harvsp', "DEFAULTSORT", #'lien',
        "ISBN", "MSAPI",  "2autres", "API",   "multiple",  
        "Titre en italique", "Titre mis en forme"
        )

    __interesting_template_name =(
        "date", "unité", "nombre", "citation", "citation bloc", "nobr", "formatnum", "pertinence détail", "pas clair", "tmp", "souverain2", 'lien'
        )

    _language_templates=("japonais", "allemand", "danois", "espagnol", "arabe", "portugais", "chinois", "biman", "thai", "hindi", "russe", "grec")

    def __init__(self, prefix='http://fr.wikipedia.org/wiki/'):
        self.prefix = prefix
        
        #1 HTML comments
        self.__comment_pattern = re.compile(r'<!--.*?-->', re.DOTALL)

        #2 HTML useless tags
        self.__garbage_tag_patterns = list()
        for tag in self.__class__.__garbage_tags:
            pattern = re.compile(
                r'<\s*%s(\s*| [^/]+?)>.*?<\s*/\s*%s\s*>' % (tag, tag), re.DOTALL | re.IGNORECASE)
            self.__garbage_tag_patterns.append(pattern)

        #3 HTML wrapper tags
        self.__wrapper_tag_patterns = list()
        for tag in self.__class__.__wrapper_tags:
            left_pattern = re.compile(
                r'<\s*%s(\s*| [^/]+?)>' % tag, re.DOTALL | re.IGNORECASE)
            right_pattern = re.compile(
                r'<\s*/\s*%s\s*>' % tag, re.DOTALL | re.IGNORECASE)
            self.__wrapper_tag_patterns.append((left_pattern, right_pattern))
        
        #4 templates and tables
        self.__good_template_patterns = list()
        for keyword in self.__class__.__interesting_template_name:
            pattern = re.compile(r"\{\{\s*%s([^{]*?)\}\}" % keyword, re.DOTALL | re.IGNORECASE)
            self.__good_template_patterns.append(pattern)

        self.__garbage_template_patterns = list()
        for keyword in self.__class__.__garbage_template_name:
            pattern2 = re.compile(r"\{\{\s*%s([^}]*?)(\{\{(.*?)\}\}(.*?))+\}\}" % keyword, re.DOTALL | re.IGNORECASE)    
            pattern = re.compile(r"\{\{\s*%s([^{]*?)\}\}" % keyword, re.DOTALL | re.IGNORECASE)
            self.__garbage_template_patterns.append(pattern)
            self.__garbage_template_patterns.append(pattern2)
        
        #5 single tags
        self.__single_tag_patterns = list()
        for tag in self.__class__.__single_tags:
            pattern = re.compile(
                r'<\s*%s(\s*| .+?)/\s*>' % tag, re.DOTALL | re.IGNORECASE)
            self.__single_tag_patterns.append(pattern)
        
        #6 Links prefixes - anchors
        self.__garbage_anchor_patterns_large = list()
        for anchor in self.__class__.__garbage_link_prefixes:
            pattern_large = re.compile(r"\[\[(\s*| [^/]+?)%s(\s*| [^/]+?):(.*?)((\[\[(.*?)\]\])(.*?))*?\]\]" % anchor, re.DOTALL | re.IGNORECASE) #[[Fichier:Noether.jpg|thumb|left|[[Emmy Noether]] utilise la notion d'espace vectoriel pour étudier les [[Anneau noethérien|anneaux]] portant maintenant son nom.]]
            self.__garbage_anchor_patterns_large.append(pattern_large)
            #use first the large patterns to remove the anchors that have a caption, therefore several layers of [[]]
            #then the restricted ones as all that will be left are "proper" [[bad_tag:something]] patterns
        
        #7 Language templates
        self.language_templates = list()
        for tag in self.__class__._language_templates:
            pattern = re.compile(
                r'\{\{%s\|([^|]*?)\|([^|]*?)\|([^|]*?)\}\}' % tag, re.DOTALL | re.IGNORECASE)
        self.language_templates.append(pattern)


    def generate_json_docs(self, input_file, number_of_workers=4, chunk_size=10000):
        for pages in self.process_dump(input_file, chunk_size):
            for page in pages :
                yield self.process_page(page)

    
    def process_dump(self, input_file, chunk_size=10000):
        pages = []
        page = []
        page_counter = 0

        for line in bz2.BZ2File(input_file, 'r'):
            line = line.decode('utf-8').strip()
            if line == '<page>':
                page = []
            elif line == '</page>':
                pages.append(page)
                page_counter += 1
                if len(pages) >= chunk_size:
                    yield pages
                    pages = []
            else:
                page.append(line)

        if len(pages) > 0:
            yield pages

    
    
    def process_page(self, page):
        """
        Always the same page structure :
        0'<title>Antoine Meillet</title>', 
        1'<ns>0</ns>', 
        2'<id>3</id>', 
        3'<revision>', 
        4'<id>158198967</id>', 
        5'<parentid>158036022</parentid>', 
        6'<timestamp>2019-04-06T11:00:52Z</timestamp>', 
        7'<contributor>', 
        8'<username>WikiCleanerBot</username>', 
        9'<id>351003</id>', 
        10'</contributor>', 
        11'<minor />', 
        12'<comment>v2.01 - [[P:CS|Correction syntaxique]] (Numéro ISBN : syntaxe erronée)</comment>',
        13'<model>wikitext</model>', 
        14'<format>text/x-wiki</format>'
        15<text> ..... 
                </text>
        
        -2'<sha1>o7mtp7p7lq3lfz1edbvmkp95ty1vyy5</sha1>', 
        -1'</revision>'
        """
        
        wiki_document = self.extract_content(page, quote=False)
        if wiki_document is None:
            return

        wiki_document = self.process_document(wiki_document)
        if wiki_document is None:
           return
            
        return (wiki_document)

    def extract_content(self,page, quote=False):
        wiki_document = AnnotatedWikiDocument()

        title = re.match(r"<title>(.*?)</title>", page[0]).group(1)
        if self.reject_page(title):
            return None
        id = re.match(r"<id>(.*?)</id>", page[2]).group(1)
        url = get_wiki_document_url(title, self.prefix, quote=quote)
        text=""

        #the text is between the <text and </text> things. 
        #bool for knowing if we are between the two text tags or not
        ##if we are, record (ie add line to the text)
        record = False
        for line in page :
            if not line:
                continue
            if line.startswith("<text"):
                if "homonymie" in line.lower():
                    return None
                elif "#redirect" in line.lower(): # anything goes along the lines of #redirect, #redirection, #REDIRECTION...
                    return None
                elif line.endswith('</text>'): # Redirect page
                    return None
                else :
                    record = True
            elif line.endswith('</text>'):
                record = False
            
            if record == True :
                text += '\n%s' % line
            else :
                continue

        wiki_document.text=text
        wiki_document.title = title
        wiki_document.url = url
        wiki_document.id = id
        return(wiki_document)

    def reject_page(self, title):
        for reject_prefix in self.__garbage_page_prefixes:
            if title.startswith(reject_prefix):
                return True
        if "abréviation en" in title.lower():
            return(True)
        elif "abréviation de" in title.lower():
            return True
        elif ("abréviation" in title.lower()) and ("liste" in title.lower()):
            return True
        elif ("chronologie" in title.lower()):
            return True
        elif (re.search(r'^[0-9]{3}[0-9]*', title.lower()) is not None): #if there is a date in the title, it is usually useless
            return True
        else:
            return False



    def process_document(self, wiki_document):
        #Clean the title a little
        wiki_document.title = wiki_document.title.replace(
            '&amp;', '&').replace('&quot;&quot;', '&quot;')
        wiki_document.title = wiki_document.title.replace('&quot;&quot;', '\'')
        wiki_document.title = wiki_document.title.replace('&nbsp;', ' ')


        #render the tags legible
        wiki_document.text = wiki_document.text.replace(
            '&lt;', '<').replace('&gt;', '>')
        wiki_document.text = wiki_document.text.replace(
            '<<', u'��').replace('>>', u'��')
        wiki_document.text = wiki_document.text.replace(
            '&amp;', '&').replace('&quot;&quot;', '&quot;')
        wiki_document.text = wiki_document.text.replace('&quot;&quot;', '\'')
        wiki_document.text = wiki_document.text.replace('&nbsp;', ' ')

        #remove the first header
        wiki_document.text = wiki_document.text.replace("<text xml:space=\"preserve\">", "")

        #remove the line breaks 
        wiki_document.text = wiki_document.text.replace("<br />", "")


        #find the categories and save them    
        match = re.findall(r"\[\[Catégorie:(.*)\]\]", wiki_document.text)
        categories =[]
        if match :
            for m in match :
                categories.append(m)
        wiki_document.categories = categories

        #remove eveything that comes after notes and references
        wiki_document.text = re.sub(r"=+\s*Notes et références\s*==(.|\s)*", "", wiki_document.text, flags= re.DOTALL)
        wiki_document.text = re.sub(r"=+\s*Voir aussi\s*==(.|\s)*", "", wiki_document.text, flags= re.DOTALL)
        wiki_document.text = re.sub(r"=+\s*Bibliographie\s*===(.|\s)*", "", wiki_document.text, flags= re.DOTALL)
        wiki_document.text = re.sub(r"=+\s*Articles connexes\s*===(.|\s)*", "", wiki_document.text, flags= re.DOTALL)
        wiki_document.text = re.sub(r"=+\s*Article connexe\s*===(.|\s)*", "", wiki_document.text, flags= re.DOTALL)
        wiki_document.text = re.sub(r"=+\s*Lien(s)? externe(s)?\s*=+(.|\s)*", "", wiki_document.text, flags= re.DOTALL)
        wiki_document.text = re.sub(r"=+\s*Lien(s)? interne(s)?\s*=+(.|\s)*", "", wiki_document.text, flags= re.DOTALL)
        #remove the three ''' that go around the title repated in article
        wiki_document.text = re.sub(r"\'\'\'","",wiki_document.text)
        
        #replace the # for lists by the usual * : # [[Jorge Altamira]] # [[Juan Carlos Arcagni]] # [[José Bonacci]]...
        wiki_document.text = re.sub(r"(^|\n)(#)(\s*|[^/]+?)\[\[(.*?)\]\]", "\\2", wiki_document.text)

        #1 remove the comments
        wiki_document.text = self.__comment_pattern.sub('', wiki_document.text)

        #3 wrapper tags
        for left_pattern, right_pattern in self.__wrapper_tag_patterns:
            wiki_document.text = left_pattern.sub('', wiki_document.text)
            wiki_document.text = right_pattern.sub('', wiki_document.text)

        #2 remove the garbage tags patterns
        for pattern in self.__garbage_tag_patterns:
            wiki_document.text = pattern.sub('', wiki_document.text)
        for left_pattern, right_pattern in self.__wrapper_tag_patterns:
            wiki_document.text = left_pattern.sub('', wiki_document.text)
            wiki_document.text = right_pattern.sub('', wiki_document.text)

        #5 remove the single tags
        for pattern in self.__single_tag_patterns:
            wiki_document.text = pattern.sub("", wiki_document.text)

        #4 remove the table templates
        for pattern in self.__garbage_template_patterns:
            wiki_document.text = pattern.sub("", wiki_document.text)

        for pattern in self.__good_template_patterns:
            for match in pattern.finditer(wiki_document.text):
                template = match.group()
                template_info = template[2:-2]
                template_info = " ".join(template_info.split("|")[1:])
                wiki_document.text = wiki_document.text.replace(
                    template, template_info)
                    #TODO : or no, it depends on how had it is to get this to work, fix stuff like "fait référence aux {{Citation|[[Levée en masse|soldats de l'an II]]}}" =>"fait référence aux Levée en masse soldats de l'an II" 



            #some very specific templates to take care of
            #the centuries ones can be not clear: we find both {{XIXe siècle}} and {{s-|XX}},  and also {{s2-|XX|e|XXI}}
            #millenaire too : {{-millénaire|III|e}}
        wiki_document.text = re.sub(r"\{\{s-\|(.*?)\}\}","\\1e siècle", wiki_document.text) 
        wiki_document.text = re.sub(r"\{\{s\|(.*?)\}\}","\\1e siècle", wiki_document.text) 

        wiki_document.text = re.sub(r"\{\{-s-\|(.*?)\}\}","\\1e siècle", wiki_document.text) 
        wiki_document.text = re.sub(r"\{\{s mini-\|(.*?)\|(.*?)\}\}","\\1e siècle", wiki_document.text) 
        wiki_document.text = re.sub(r"\{\{s2-\|(.*?)\|(.*?)\|(.*?)\}\}", "\\1e-\\2e siècles", wiki_document.text)
        wiki_document.text = re.sub(r"\{\{s2\|(.*?)\|(.*?)\|(.*?)\}\}", "\\1e-\\2e siècles", wiki_document.text)
        wiki_document.text = re.sub(r"\{\{sp-\|(.*?)\|e\|et\|(.*?)\|e\|s\}\}", "\\1e-\\2e siècles", wiki_document.text)

        wiki_document.text = re.sub(r"\{\{-millénaire\|(.*?)\|*(.*?)\}\}", "\\1\\2 millénaire", wiki_document.text)   

            #the language ones : {{lang|en|''[[Mémoire virtuelle#Swapping|swapping]]''}}
        wiki_document.text = re.sub(r"\{\{lang(ue)?\|([^}]*?)\|([^}]*?)\}\}","\\3", wiki_document.text) 
        wiki_document.text = re.sub(r"\{\{lang-([^}]*?)\|([^}]*?)\}\}","\\2", wiki_document.text) 
        wiki_document.text = re.sub(r"\{\{Langue\|([^}]*?)\|([^}]*?)\}\}","\\2", wiki_document.text) 
        wiki_document.text = re.sub(r"\{\{Langue\|([^}]*?)\|([^}]*?)\|([^}]*?)\}\}","\\3", wiki_document.text) 
        for pattern in self.language_templates:
            wiki_document.text = re.sub(pattern,"\\3", wiki_document.text) #in the languages, we take the "translitteration" of foreign names in french
        
            #the weird ones : {{…}}
        wiki_document.text = re.sub(r"\{\{…\}\}", "", wiki_document.text)
        wiki_document.text = re.sub(r"\{\{...\}\}", "", wiki_document.text)
        
            #if we still have a {{}} kind of template, consider it just styling : we simply remove the {{ and }}. We do it in two folds, in case we have {{}} inside some other {{}}
        wiki_document.text = re.sub(r"\{\{(?!Infobox)([^|{]*?)\}\}", "\\1", wiki_document.text) 
        wiki_document.text = re.sub(r"\{\{(?!Infobox)([^|}]*?)\|([^|{]*?)(\|([^|{]*?))*\}\}", "\\2", wiki_document.text)
        wiki_document.text = re.sub(r"\{\{(?!Infobox)([^|}]*?)\|([^|{]*?)(\|([^|{]*?))*\}\}", "\\2", wiki_document.text)
        #wiki_document.text = re.sub(r"\{\{(?!Infobox)([^}]*?)(\{\{(.*?)\}\}(.*?))+\}\}", "\\1", wiki_document.text)

        #print("After infobox")

        #remove the infobox
        wiki_document.text = re.sub(r"\{\{Infobox\s*(\s*?[^{]*?)\n\}\}", "", wiki_document.text, flags = re.DOTALL |re.IGNORECASE ) #for normal infoboxes
        wiki_document.text = re.sub(r"\{\{Infobox\s*(\s*?[^{]*?)\}\}", "", wiki_document.text, flags = re.DOTALL |re.IGNORECASE ) #for one-liner infoboxes inside a table (nightmares are made of nested infoboxes)
            
        wiki_document.text = re.sub(r"\{\{infobox\s*(\s*?[^{]*?)\n\}\}", "", wiki_document.text, flags = re.DOTALL |re.IGNORECASE ) #because some dude thought it'd be funny to write it without the capital I

        #remove the wikitables
        wiki_document.text = re.sub(r"(\\\w+\{[^{}]*?\})[^}]","MATH_FORMULA", wiki_document.text) #remove the math formulas beforehand because they have annoying {}
        wiki_document.text = re.sub(r"\{\|([^{}]*?)(\{\|([^{}]*?)\|\})*?([^{}]*?)\|\}", "", wiki_document.text) #first tables
        wiki_document.text = re.sub(r"\{\|([^{}]*?)(\{\|([^{}]*?)\|\})*?([^{}]*?)\|\}\n*(.*?source.*)", "", wiki_document.text, flags = re.IGNORECASE) #a second time for nested tables (hell exists and it's made of wikitables...)
        

        #6 Deal with the garbage links [[]]
        for pattern in self.__garbage_anchor_patterns_large:
            wiki_document.text = pattern.sub("", wiki_document.text)


        #remove the paragraph titles
        wiki_document.text = re.sub(r"=+\s*?(.*?)\s*?=+","\n\\1.\r\n",wiki_document.text)

        #remove the painful lists
        wiki_document.text = re.sub(r"(\*)+\s*\[([^\[\]]*?)\]\s*([^\n]*)", "", wiki_document.text, flags = re.DOTALL | re.IGNORECASE) #* [[Propriétés métriques des droites et plans]] 
        #wiki_document.text = re.sub(r"(\*)+\s*\[(.*)\](.*)?", "", wiki_document.text, flags = re.DOTALL | re.IGNORECASE) #wiki_document.text = re.sub(r"(\*)+ \[\[(.*)\]\](.*)?", "", wiki_document.text)
        #wiki_document.text = re.sub(r"(\*)+\s*\{\{(.*)\}\}\s(.*)?", "", wiki_document.text, flags = re.DOTALL | re.IGNORECASE) #* {{en}} [http://www.egwald.com/linearalgebra/index.php Linear Algebra] par Elmer G. Wiens
        #wiki_document.text = re.sub(r"(\*)+\s*(.*?)\'\'(.*?)\'\'\s*(.*)", "", wiki_document.text, flags = re.DOTALL | re.IGNORECASE)

        #clear trailing whitespace
        wiki_document.text = wiki_document.text.replace('&quot;', '\'')
        wiki_document.text = re.sub(r"\*\n", "", wiki_document.text)
        wiki_document.text = wiki_document.text.strip()


        #then now deal with the real anchors
        anchor_m = re.finditer(r"\[\[(.*?)\]\]", wiki_document.text)
        
        delta = 0
        annotations = []
        for anch in anchor_m:
            labels = anch.group().split("|")
            surface_form = labels[-1] #this way, if the page link and surface form are the same no error is thrown and we have the right ones
            page_link = labels[0]
            match = anch.group()
            offset = anch.start() - delta
            annotations.append({"uri": get_wiki_document_url(page_link, self.prefix, quote=True), "surface_form": surface_form, "offset":offset}) #we remove 2 because of the square brackets
            delta = delta + 4 + len(page_link)+1 if (len(labels) == 2) else delta+4

        wiki_document.text = re.sub(r"\[\[([^\[\|]*?)\]\]", "\\1", wiki_document.text)
        wiki_document.text = re.sub(r"\[\[([^\[]*?)\|(.*?)\]\]", "\\2", wiki_document.text)
        # add annotation to document
        wiki_document.annotations = annotations
        
        return(wiki_document)

        
if __name__ == "__main__":
    gen = WikiExtractor().generate_json_docs("/home/cyrielle/Codes/brouillons/frwiki-20190701-pages-articles-multistream1.xml-p3p275787.bz2")
    for i in range(0, 31):
        next(gen)


