class Helper:
    def chomp_new_line(self, string):
        if string.endswith('\r\n'): 
            return string[:-2]
        if string.endswith('\n') or string.endswith('\r'): 
            return string[:-1]
