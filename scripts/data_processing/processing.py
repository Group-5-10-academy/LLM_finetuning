import re
class normalize(object):
    expansion_file_dir='' # assume you have file with list of short forms with their expansion as gazeter
    short_form_dict={}
    # Constructor
    def __init__(self):
       self.short_form_dict=self.get_short_forms()
        

    def get_short_forms(self):
         text=open(self.expansion_file_dir,encoding='utf8')
         exp={}
         for line in iter(text):
            line=line.strip()
            if not line:  # line is blank
                continue
            else:
                expanded=line.split("-")
                exp[expanded[0].strip()]=expanded[1].replace(" ",'_').strip()
         return exp
                     

    # method that expand short form file
    def expand_short_form(self,input_short_word):
        if input_short_word in self.short_form_dict:              
            return self.short_form_dict[input_short_word]
        else:
            return input_short_word

#replacing any existance of special character or punctuation to null Amharic puncutation marks: =፡።፤;፦፧፨፠፣

    def remove_punc_and_special_chars(text): 
        normalized_text = re.sub('[\!\@\#\$\%\^\«\»\&\*\(\)\…\[\]\{\}\;\“\”\›\’\‘\"\'\:\,\.\‹\/\<\>\?\\\\|\`\´\~\-\=\   +\፡\።\፤\;\፦\፥\፧\፨\፠\፣]', '',text) 
        return normalized_text
#remove all ascii characters and Arabic and Amharic numbers
    def remove_ascii_and_numbers(text_input):
        rm_num_and_ascii=re.sub('[A-Za-z0-9]','',text_input)
        return re.sub('[\'\u1369-\u137C\']+','',rm_num_and_ascii)
    

#The following function performs character level mismatch normalization task. Amharic has different characters that are interchangeably used in writing and reading such as (ሀ, ኀ, ሐ, and ኸ), (ሰ and ሠ), (ጸ and ፀ), (ው and ዉ) and (አ and ዓ). For example, ጸሀይ to mean sun can also be written as ፀሐይ. In addition, Amharic words with suffix such as ቷል are also written as ቱዋል. 

    def normalize_char_level_missmatch(input_token):
        rep1=re.sub('[ሃኅኃሐሓኻ]','ሀ',input_token)
        rep2=re.sub('[ሑኁዅ]','ሁ',rep1)
        rep3=re.sub('[ኂሒኺ]','ሂ',rep2)
        rep4=re.sub('[ኌሔዄ]','ሄ',rep3)
        rep5=re.sub('[ሕኅ]','ህ',rep4)
        rep6=re.sub('[ኆሖኾ]','ሆ',rep5)
        rep7=re.sub('[ሠ]','ሰ',rep6)
        rep8=re.sub('[ሡ]','ሱ',rep7)
        rep9=re.sub('[ሢ]','ሲ',rep8)
        rep10=re.sub('[ሣ]','ሳ',rep9)
        rep11=re.sub('[ሤ]','ሴ',rep10)
        rep12=re.sub('[ሥ]','ስ',rep11)
        rep13=re.sub('[ሦ]','ሶ',rep12)
        rep14=re.sub('[ዓኣዐ]','አ',rep13)
        rep15=re.sub('[ዑ]','ኡ',rep14)
        rep16=re.sub('[ዒ]','ኢ',rep15)
        rep17=re.sub('[ዔ]','ኤ',rep16)
        rep18=re.sub('[ዕ]','እ',rep17)
        rep19=re.sub('[ዖ]','ኦ',rep18)
        rep20=re.sub('[ጸ]','ፀ',rep19)
        rep21=re.sub('[ጹ]','ፁ',rep20)
        rep22=re.sub('[ጺ]','ፂ',rep21)
        rep23=re.sub('[ጻ]','ፃ',rep22)
        rep24=re.sub('[ጼ]','ፄ',rep23)
        rep25=re.sub('[ጽ]','ፅ',rep24)
        rep26=re.sub('[ጾ]','ፆ',rep25)
        #Normalizing words with Labialized Amharic characters such as በልቱዋል or  በልቱአል to  በልቷል  
        rep27=re.sub('(ሉ[ዋአ])','ሏ',rep26)
        rep28=re.sub('(ሙ[ዋአ])','ሟ',rep27)
        rep29=re.sub('(ቱ[ዋአ])','ቷ',rep28)
        rep30=re.sub('(ሩ[ዋአ])','ሯ',rep29)
        rep31=re.sub('(ሱ[ዋአ])','ሷ',rep30)
        rep32=re.sub('(ሹ[ዋአ])','ሿ',rep31)
        rep33=re.sub('(ቁ[ዋአ])','ቋ',rep32)
        rep34=re.sub('(ቡ[ዋአ])','ቧ',rep33)
        rep35=re.sub('(ቹ[ዋአ])','ቿ',rep34)
        rep36=re.sub('(ሁ[ዋአ])','ኋ',rep35)
        rep37=re.sub('(ኑ[ዋአ])','ኗ',rep36)
        rep38=re.sub('(ኙ[ዋአ])','ኟ',rep37)
        rep39=re.sub('(ኩ[ዋአ])','ኳ',rep38)
        rep40=re.sub('(ዙ[ዋአ])','ዟ',rep39)
        rep41=re.sub('(ጉ[ዋአ])','ጓ',rep40)
        rep42=re.sub('(ደ[ዋአ])','ዷ',rep41)
        rep43=re.sub('(ጡ[ዋአ])','ጧ',rep42)
        rep44=re.sub('(ጩ[ዋአ])','ጯ',rep43)
        rep45=re.sub('(ጹ[ዋአ])','ጿ',rep44)
        rep46=re.sub('(ፉ[ዋአ])','ፏ',rep45)
        rep47=re.sub('[ቊ]','ቁ',rep46) #ቁ can be written as ቊ
        rep48=re.sub('[ኵ]','ኩ',rep47) #ኩ can be also written as ኵ  
        
        return rep48
    


#Normalize Geez and Arabic Number Mismatch

#This code snippet allows you to expand decimal form numbers to text representation. It also automatically normalize arabic numbers to Geez form. For example, 1=፩, 2=፪, …

    def arabic2geez(arabicNumber):
        ETHIOPIC_ONE= 0x1369
        ETHIOPIC_TEN= 0x1372
        ETHIOPIC_HUNDRED= 0x137B
        ETHIOPIC_TEN_THOUSAND = 0x137C
        arabicNumber=str(arabicNumber)
        n = len(arabicNumber)-1 #length of arabic number
        if n%2 == 0:
            arabicNumber = "0" + arabicNumber
            n+=1
        arabicBigrams=[arabicNumber[i:i+2] for i in range(0,n,2)] #spliting bigrams
        reversedArabic=arabicBigrams[::-1] #reversing list content
        geez=[]
        for index,pair in enumerate(reversedArabic):
            curr_geez=''
            artens=pair[0]#arrabic tens
            arones=pair[1]#arrabic ones
            amtens=''
            amones=''
            if artens!='0':
                amtens=str(chr((int(artens) + (ETHIOPIC_TEN - 1)))) #replacing with Geez 10s [፲,፳,፴, ...]
            else:
                if arones=='0': #for 00 cases
                    continue
            if arones!='0':       
                    amones=str(chr((int(arones) + (ETHIOPIC_ONE - 1)))) #replacing with Geez Ones [፩,፪,፫, ...]
            if index>0:
                if index%2!= 0: #odd index
                    curr_geez=amtens+amones+ str(chr(ETHIOPIC_HUNDRED)) #appending ፻
                else: #even index
                    curr_geez=amtens+amones+ str(chr(ETHIOPIC_TEN_THOUSAND)) # appending ፼
            else: #last bigram (right most part)
                curr_geez=amtens+amones
            
            geez.append(curr_geez)
        
        geez=''.join(geez[::-1])
        if geez.startswith('፩፻') or geez.startswith('፩፼'):
            geez=geez[1:]
        
        if len(arabicNumber)>=7:
            end_zeros=''.join(re.findall('([0]+)$',arabicNumber)[0:])
            i=int(len(end_zeros)/3)
            if len(end_zeros)>=(3*i):
                if i>=3:
                    i-=1
                for thoushand in range(i-1):
                    print(thoushand)                
                    geez+='፼'

        return geez
    
    def getExpandedNumber(self,number):
        if '.' not in str(number):
            return arabic2geez(number)
        else:
            num,decimal=str(number).split('.')
            if decimal.startswith('0'):
                decimal=decimal[1:]
                dot=' ነጥብ ዜሮ '
            else:
                dot=' ነጥብ '
            return arabic2geez(num)+dot+self.arabic2geez(decimal)

    