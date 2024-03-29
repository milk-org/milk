# Readline Input Keys {#page_readlinekeys}


The command line interface (CLI) uses readline to take input command for keyboard. Readline offers convenient bindings, listed in this page.

For more information, consult the readline library documentation.


<pre>

READLINE QUICK HELP


Notations:
C- is control key
M- is meta key (if already bound to action, use ESC and then type the folling character)

C-_ or C-x C-u    Undo the last editing command. You can undo all the way back to an empty line.
C-a	    Move to the start of the line
C-e         Move to the end of the line
M-f         Move forward a word, where a word is composed of letters and digits.
M-b         Move backward a word.
C-l         Clear the screen, reprinting the current line at the top.

C-k         Kill the text from the current cursor position to the end of the line.
M-d         Kill from the cursor to the end of the current word, or, if between words, to the end of the next word.
M-DEL       Kill from the cursor the start of the current word, or, if between words, to the start of the previous
C-w         Kill from the cursor to the previous whitespace. This is different than M-DEL because the word boundaries differ.

C-y         Yank the most recently killed text back into the buffer at the cursor.
M-y         Rotate the kill-ring, and yank the new top. You can only do this if the prior command is C-y or ESC y.

M-<         Move to the first line in the history.
M->         Move to the end of the input history, i.e., the line currently being entered.

C-t	    Drag the character before the cursor forward over the character at the cursor, moving the cursor forward as well. If the insertion point is at the end of the line, then this transposes the last two characters of the line. Negative arguments have no effect.
M-t	    Drag the word before point past the word after point, moving point past that word as well. If the insertion point is at the end of the line, this transposes the last two words on the line.
M-u	    Uppercase the current (or following) word. With a negative argument, uppercase the previous word, but do not move the cursor.
M-l	    Lowercase the current (or following) word. With a negative argument, lowercase the previous word, but do not move the cursor.
M-c	    Capitalize the current (or following) word. With a negative argument, capitalize the previous word, but do not move the cursor.

TAB         Attempt to perform completion on the text before point. The actual completion performed is application-specific. The default is filename completion.
M-?         List the possible completions of the text before point.
M-*         Insert all completions of the text before point that would have been generated by possible-completions.

</pre>
