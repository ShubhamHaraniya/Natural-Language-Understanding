import re
from datetime import date

def calculate_age(day, month, year):
    """Calculates age given birth day, month and year."""
    today = date.today()
    
    # handle 2-digit years - people born in '99 could be 1999 or way back in 1899
    if year < 100:
        # if it's below 25, probably 2000s. otherwise must be 1900s
        # nobody alive from 1800s anyway
        if year < 25:
            year += 2000
        else:
            year += 1900
            
    try:
        birth_date = date(year, month, day)
    except ValueError:
        return None  # couldn't make a valid date from this
        
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    return age

def parse_date(date_str):
    """Parses date string using regex and returns age."""
    
    # first try to match numeric dates like 12-05-1999 or 12/05/1999
    # supports dd-mm-yyyy, mm-dd-yyyy with slashes, dashes, or dots
    # figuring out which is day vs month is tricky though
    numeric_match = re.search(r'\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})\b', date_str)
    if numeric_match:
        p1, p2, p3 = map(int, numeric_match.groups())
        
        # okay so here's the logic: if one number is > 12, that's gotta be the day
        # if p1 > 12, must be dd-mm-yyyy format
        # if p2 > 12, must be mm-dd-yyyy format (like 12-25-2000)
        # if both are <= 12, it's ambiguous. we'll just assume dd-mm-yyyy
        # since that's more common outside the US
        
        day, month, year = 0, 0, p3
        
        if p2 > 12:  # something like 12-25-2000, so mm-dd-yyyy
            month, day = p1, p2
        elif p1 > 12:  # something like 25-12-2000, so dd-mm-yyyy
            day, month = p1, p2
        else:
            # both numbers work as day or month, e.g., 05-06-1990
            # just gonna assume dd-mm-yyyy here to be safe
            day, month = p1, p2
            
        return calculate_age(day, month, year)

    # now try text-based dates like "15 August 1995" or "15 Aug 95"
    text_match = re.search(r'\b(\d{1,2})[\s\t]+([a-zA-Z]+)[\s\t,]+(\d{2,4})\b', date_str, re.I)
    if text_match:
        d, m_str, y = text_match.groups()
        day = int(d)
        year = int(y)
        
        # convert month names to numbers
        months = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        m_str = m_str.lower()[:3]  # just take first 3 letters so "August" -> "aug"
        if m_str in months:
            return calculate_age(day, months[m_str], year)

    return None

def chatbot():
    print("Reggy: Hello! I am Reggy++. I can predict your age and detect your mood.")
    
    # keeping track of what info we've collected so far
    name_found = False
    age_found = False
    mood_found = False
    
    user_name = ""
    user_surname = ""
    
    while True:
        try:
            prompt = "You: "
            user_input = input(prompt).strip()
        except EOFError:
            break
            
        if not user_input:
            continue
            
        # check if user wants to leave
        if re.search(r'\b(bye|exit|quit|stop)\b', user_input, re.I):
            print(f"Reggy: Goodbye {user_name}! Have a nice day.")
            break
            
        # Step 1: Get their name first
        if not name_found:
            # looking for stuff like "My name is John Doe" or "I am John"
            # need the full name so we can extract surname later
            match = re.search(r"(?:name is|i am|i'm|this is)\s+([a-zA-Z\s\.]+)", user_input, re.I)
            
            full_name = ""
            if match:
                full_name = match.group(1).strip()
            # if no clear pattern, maybe they just typed their name directly?
            # but don't want to grab "Hello" as a name, so filter those out
            elif re.match(r"^[a-zA-Z\s]+$", user_input) and len(user_input.split()) <= 3 and not re.search(r'\b(hello|hi|hey)\b', user_input, re.I):
                full_name = user_input.strip()
            
            if full_name:
                name_parts = full_name.split()
                # clean up any filler words we accidentally captured
                name_parts = [p for p in name_parts if p.lower() not in ['my', 'name', 'is', 'i', 'am']]
                
                if not name_parts:
                    print("Reggy: I didn't catch that. What is your name?")
                    continue

                user_name = name_parts[0]  # grab the first name
                if len(name_parts) > 1:
                    user_surname = name_parts[-1]  # surname is usually the last part
                
                print(f"Reggy: Nice to meet you, {user_name}!" + (f" (Or should I say Mr./Ms. {user_surname}?)" if user_surname else ""))
                print("Reggy: When is your birthday? (e.g., DD-MM-YYYY)")
                name_found = True
            else:
                # they said hi but didn't give a name
                if re.search(r'\b(hello|hi|hey)\b', user_input, re.I):
                    print("Reggy: Hi there! What's your name?")
                else:
                    print("Reggy: Could you tell me your name first?")
            continue
            
        # Step 2: Ask for birthday and calculate age
        if not age_found:
            age = parse_date(user_input)
            if age is not None:
                print(f"Reggy: Based on your birthday, you are currently {age} years old.")
                print(f"Reggy: How are you feeling today, {user_name}?")
                age_found = True
            else:
                print("Reggy: I couldn't understand that date format. Please try something like 12-05-1999 or 12 May 1999.")
            continue
            
        # Step 3: Check their mood
        if not mood_found:
            # trying to catch positive/negative moods even with typos
            # using + to allow repeated letters like "soooo happyyy"
            
            is_positive = re.search(r'\b(g+o+d+|h+a+p+y+|g+r+e+a*t+|f+i+n+e+|o+k+|w+e+l+)\b', user_input, re.I)
            
            # same thing for negative moods
            is_negative = re.search(r'\b(s+a+d+|b+a+d+|t+e+r+i+b+l+e*|t+i+r+e+d+|b+o+r+e+d*)\b', user_input, re.I)
            
            if is_positive:
                print("Reggy: That's awesome! Keep smiling!")
                print("Reggy: (Type 'bye' to exit or tell me more)")
                mood_found = True
            elif is_negative:
                print("Reggy: Oh no, I hope your day gets better. Sending virtual hugs!")
                print("Reggy: (Type 'bye' to exit or tell me more)")
                mood_found = True
            else:
                print("Reggy: I see. Can you tell me more about how you feel? (I might have trouble understanding complex emotions!)")
        else:
            # mood already detected, just keep chatting
            print(f"Reggy: Tell me more!")

if __name__ == "__main__":
    chatbot()
