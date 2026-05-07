from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """ This is the text which i will work on.

Isn't there any library in python which can do lorem ipsum like i did once.

Ohh, probably the 2nd sentence was a bit too big for the splitter.

And same goes for the 3rd one. 

But now that this is the last line, I can go all out and type whatever I want. Man i like typing on my own but something's keeping me from doing this. I am going crazy over this; maybe i should stop talking to myself."""

print("\n\n========Here comes our shinning star, RCTS========")

splitter = RecursiveCharacterTextSplitter(
    separators = ["\n\n","\n",". "," ",""],
    chunk_size = 100,
    chunk_overlap = 0
)

chunks = splitter.split_text(text)
print("\nPrinting chunks with RCTS\n")
for i,chunk in enumerate(chunks):
    print(f'Chunk {i+1}: {len(chunk)} characters')
    print(f'"{chunk}"')
    print()