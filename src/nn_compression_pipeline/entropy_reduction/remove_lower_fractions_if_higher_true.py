from .float_remove_sections_if_chosen_bits_set import FloatRemoveSectionsIfChosenBitsSet


class RemoveLowerFractionsIfHigherTrue(FloatRemoveSectionsIfChosenBitsSet):
    """
        due to buildup of np.single per platform, platform dependent
    """
    lossy = True

    # testing = True

    def __init__(self, numb_start_bits_to_check: int, bits_to_remove_at_end: int):
        self.numb_start_bits_to_check = numb_start_bits_to_check
        self.bits_to_remove_at_end = bits_to_remove_at_end

        numb_bits_fraction = 23
        total_bits = 32

        start_cut = numb_bits_fraction - self.numb_start_bits_to_check

        super().__init__(start_cut, numb_bits_fraction, self.bits_to_remove_at_end, total_bits)

    def algorithm_name(self, dic_params=None):
        if dic_params:
            return super().algorithm_name(dic_params)

        return super().algorithm_name({
            'sbc': self.numb_start_bits_to_check,
            'bre': self.bits_to_remove_at_end
        })
